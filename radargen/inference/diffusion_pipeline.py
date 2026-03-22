# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import dataclass, field
from typing import Optional

import pyrallis
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")  # ignore warning

import numpy as np
from einops import rearrange
from PIL import Image

from diffusion import DPMS, FlowEuler
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode
from diffusion.model.utils import get_weight_dtype
from diffusion.model_modification import inflate_sana_input_channels_for_radargen
from diffusion.utils.config import SanaConfig, model_init_config
from diffusion.utils.logger import get_root_logger
from radargen.inference.model_download import find_model
from radargen.radar_maps import (
    decoded_to_doppler_map,
    decoded_to_pd_map,
    decoded_to_rcs_map,
    doppler_map_to_image,
    pd_map_to_image,
    rcs_map_to_image,
)


def guidance_type_select(default_guidance_type, pag_scale, attn_type):
    guidance_type = default_guidance_type
    if not (pag_scale > 1.0 and attn_type == "linear"):
        guidance_type = "classifier-free"
    elif pag_scale > 1.0 and attn_type == "linear":
        guidance_type = "classifier-free_PAG"
    return guidance_type


@dataclass
class RadarGenOutput:
    pd_map_np: np.ndarray
    pd_img_pil: Image.Image
    rcs_map_np: np.ndarray
    rcs_img_pil: Image.Image
    doppler_map_np: np.ndarray
    doppler_img_pil: Image.Image


@dataclass
class RadarGenInferenceArgs(SanaConfig):
    config: Optional[str] = "configs/RadarGen_600M_512px_TS_inference.yaml"
    model_path: str = field(
        default="output/RadarGen_600M_512px_TS/RadarGen_600M_512px_TS.yaml", metadata={"help": "Path to the model file (positional)"}
    )
    output: str = "./output"
    bs: int = 1
    image_size: int = 512
    cfg_scale: float = 0.0
    pag_scale: float = 0.0
    seed: int = 42
    step: int = -1


class RadarGenDiffusionPipeline(nn.Module):
    def __init__(
        self,
        config: Optional[str] = "configs/RadarGen_600M_512px_TS_inference.yaml",
        model: Optional[nn.Module] = None,
        vae: Optional[nn.Module] = None,
        tokenizer: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        num_conditions: Optional[int] = None,
    ):
        super().__init__()
        config = pyrallis.load(RadarGenInferenceArgs, open(config))
        self.args = self.config = config
        extra = config.data.extra
        self.rcs_min = extra['rcs_min']
        self.rcs_max = extra['rcs_max']
        self.doppler_min = extra['doppler_min']
        self.doppler_max = extra['doppler_max']

        # set some hyper-parameters
        self.image_size = self.config.model.image_size

        self.num_maps = config.model.extra['num_maps']
        self.num_conditions = config.model.extra['num_conditions'] if num_conditions is None else num_conditions

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        logger = get_root_logger()
        self.logger = logger
        self.progress_fn = lambda progress, desc: None

        self.latent_size = self.image_size // config.vae.vae_downsample_rate
        self.max_sequence_length = config.text_encoder.model_max_length
        self.flow_shift = config.scheduler.flow_shift
        guidance_type = "classifier-free_PAG"

        weight_dtype = get_weight_dtype(config.model.mixed_precision)
        self.weight_dtype = weight_dtype
        self.vae_dtype = torch.float16 #get_weight_dtype(config.vae.weight_dtype)

        # self.base_ratios = eval(f"ASPECT_RATIO_{self.image_size}_TEST")
        self.vis_sampler = self.config.scheduler.vis_sampler
        logger.info(f"Sampler {self.vis_sampler}, flow_shift: {self.flow_shift}")
        self.guidance_type = guidance_type_select(guidance_type, self.args.pag_scale, config.model.attn_type)
        logger.info(f"Inference with {self.weight_dtype}, PAG guidance layer: {self.config.model.pag_applied_layers}")

        # 1. build vae and text encoder
        self.vae = self.build_vae(config.vae) if vae is None else vae
        self.tokenizer, self.text_encoder = self.build_text_encoder(config.text_encoder) if tokenizer is None or text_encoder is None else (tokenizer, text_encoder)

        if model is None:
            # Build Sana model if not provided
            self.model = self.build_sana_model(config).to(self.device)
            inflate_sana_input_channels_for_radargen(self.model, logger, num_conditions=self.num_conditions)
        else:
            # Use the provided RadarGen model
            self.model = model

        # 3. pre-compute null embedding
        with torch.no_grad():
            null_caption_token = self.tokenizer(
                "", max_length=self.max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(self.device)
            self.null_caption_embs = self.text_encoder(null_caption_token.input_ids, null_caption_token.attention_mask)[
                0
            ]

    def build_vae(self, config):
        vae = get_vae(config.vae_type, config.vae_pretrained, self.device).to(self.vae_dtype)
        return vae

    def build_text_encoder(self, config):
        tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder_name, device=self.device)
        return tokenizer, text_encoder

    def build_sana_model(self, config):
        # model setting
        model_kwargs = model_init_config(config, latent_size=self.latent_size)
        model_kwargs['num_maps'] = self.num_maps
        model_kwargs['num_conditions'] = self.num_conditions
        model = build_model(
            config.model.model,
            use_fp32_attention=config.model.get("fp32_attention", False) and config.model.mixed_precision != "bf16",
            **model_kwargs,
        )
        self.logger.info(f"use_fp32_attention: {model.fp32_attention}")
        self.logger.info(
            f"{model.__class__.__name__}:{config.model.model},"
            f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}"
        )
        return model

    def from_pretrained(self, model_path):
        state_dict = find_model(model_path)
        state_dict = state_dict.get("state_dict", state_dict)
        if "pos_embed" in state_dict:
            del state_dict["pos_embed"]
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        self.model.eval().to(self.weight_dtype)

        self.logger.info("Generating sample from ckpt: %s" % model_path)
        self.logger.warning(f"Missing keys: {missing}")
        self.logger.warning(f"Unexpected keys: {unexpected}")

    def register_progress_bar(self, progress_fn=None):
        self.progress_fn = progress_fn if progress_fn is not None else self.progress_fn

    @torch.inference_mode()
    def forward(
        self,
        prompt=None,
        height_single_image=512,
        width_single_image=512,
        height_num_images=1,
        width_num_images=1,
        negative_prompt="",
        num_inference_steps=20,
        guidance_scale=0.0,
        pag_guidance_scale=0.0,
        num_images_per_prompt=1,
        generator=torch.Generator().manual_seed(42),
        latents=None,
        bev_condition_maps=None,
    ):
        if prompt is not None and prompt != "":
            raise ValueError("prompt argument is not supported for RadarGen model")
        
        if bev_condition_maps is None:
            raise ValueError("bev_condition_maps argument is required for RadarGen model")
        else:
            with torch.amp.autocast(
                "cuda",
                enabled=(self.config.model.mixed_precision == "fp16" or self.config.model.mixed_precision == "bf16"),
            ):
                for i in range(len(bev_condition_maps)):
                    bev_condition_maps[i] = bev_condition_maps[i].to(self.device)
                    bev_condition_maps[i] = vae_encode(self.config.vae.vae_type, self.vae, bev_condition_maps[i], self.config.vae.sample_posterior, self.device)

        self.latent_size_h, self.latent_size_w = (
            (height_num_images * height_single_image) // self.config.vae.vae_downsample_rate,
            (width_num_images * width_single_image) // self.config.vae.vae_downsample_rate,
        )
        self.guidance_type = guidance_type_select(self.guidance_type, pag_guidance_scale, self.config.model.attn_type)

        # 1. pre-compute negative embedding
        if negative_prompt != "":
            null_caption_token = self.tokenizer(
                negative_prompt,
                max_length=self.max_sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            self.null_caption_embs = self.text_encoder(null_caption_token.input_ids, null_caption_token.attention_mask)[
                0
            ]

        if prompt is None:
            prompt = [""]
        prompts = prompt if isinstance(prompt, list) else [prompt]
        samples = []

        for prompt in prompts:
            # data prepare
            hw = torch.tensor([[height_single_image, width_single_image]], dtype=torch.float, device=self.device).repeat(
                num_images_per_prompt, 1
            )
            ar = torch.tensor([[1.0]], device=self.device).repeat(num_images_per_prompt, 1)
            
            with torch.no_grad():
                max_length_all = self.config.text_encoder.model_max_length

                caption_token = self.tokenizer(
                    [prompt],
                    max_length=max_length_all,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(device=self.device)
                select_index = [0] + list(range(-self.config.text_encoder.model_max_length + 1, 0))
                caption_embs = self.text_encoder(caption_token.input_ids, caption_token.attention_mask)[0][:, None][
                    :, :, select_index
                ].to(self.weight_dtype)
                emb_masks = caption_token.attention_mask[:, select_index]
                null_y = self.null_caption_embs.repeat(len(prompts), 1, 1)[:, None].to(self.weight_dtype)

                n = len(prompts)
                if latents is None:
                    z = torch.randn(
                        n * self.num_maps,
                        self.config.vae.vae_latent_dim,
                        self.latent_size_h,
                        self.latent_size_w,
                        generator=generator,
                        device=self.device,
                    )
                else:
                    z = latents.to(self.device)
                
                caption_embs = torch.cat([caption_embs] * self.num_maps, dim=0) # Repeat the caption embeddings for each map
                null_y = torch.cat([null_y] * self.num_maps, dim=0) # Repeat the null embeddings for each map

                emb_masks = emb_masks.unsqueeze(0).unsqueeze(0) # (1, 300) -> (1, 1, 1, 300), where the first dimension is batch size
                emb_masks = emb_masks.repeat(n * self.num_maps, 1, 1, 1) # Because it's batch size, we need to repeat the mask for each map
                emb_masks = torch.cat([emb_masks] * 2, dim=0) # 2 for classifier-free guidance
                model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=emb_masks)
                
                model_kwargs['bev_condition_maps'] = bev_condition_maps

                if self.vis_sampler == "flow_euler":
                    flow_solver = FlowEuler(
                        self.model,
                        condition=caption_embs,
                        uncondition=null_y,
                        cfg_scale=guidance_scale,
                        model_kwargs=model_kwargs,
                    )
                    sample = flow_solver.sample(
                        z,
                        steps=num_inference_steps,
                    )
                elif self.vis_sampler == "flow_dpm-solver":
                    scheduler = DPMS(
                        self.model,
                        condition=caption_embs,
                        uncondition=null_y,
                        guidance_type=self.guidance_type,
                        cfg_scale=guidance_scale,
                        pag_scale=pag_guidance_scale,
                        pag_applied_layers=self.config.model.pag_applied_layers,
                        model_type="flow",
                        model_kwargs=model_kwargs,
                        schedule="FLOW",
                    )
                    scheduler.register_progress_bar(self.progress_fn)
                    sample = scheduler.sample(
                        z,
                        steps=num_inference_steps,
                        order=2,
                        skip_type="time_uniform_flow",
                        method="multistep",
                        flow_shift=self.flow_shift,
                    )

            sample = sample.to(self.vae_dtype)
            with torch.no_grad():
                z = rearrange(sample, '(b v) c h w -> b v c h w', v=self.num_maps).to(torch.float16)
                latent_pd_map = z[:, 0]
                latent_rcs_map = z[:, 1]
                latent_doppler_map = z[:, 2]
                
                decoded_pd_map = vae_decode(self.config.vae.vae_type, self.vae, latent_pd_map)
                decoded_pd_map = torch.clamp(127.5 * decoded_pd_map + 128.0, 0, 255).permute(0, 2, 3, 1).detach().to("cpu", dtype=torch.float32).numpy()
                pd_map_np = decoded_to_pd_map(decoded_pd_map[0])
                pd_img_pil, _ = pd_map_to_image(pd_map_np)

                decoded_rcs_map = vae_decode(self.config.vae.vae_type, self.vae, latent_rcs_map)
                decoded_rcs_map = torch.clamp(127.5 * decoded_rcs_map + 128.0, 0, 255).permute(0, 2, 3, 1).detach().to("cpu", dtype=torch.float32).numpy()
                rcs_map_np = decoded_to_rcs_map(decoded_rcs_map[0], rcs_min=self.rcs_min, rcs_max=self.rcs_max)
                rcs_img_pil, _ = rcs_map_to_image(rcs_map_np, rcs_min=self.rcs_min, rcs_max=self.rcs_max)

                decoded_doppler_map = vae_decode(self.config.vae.vae_type, self.vae, latent_doppler_map)
                decoded_doppler_map = torch.clamp(127.5 * decoded_doppler_map + 128.0, 0, 255).permute(0, 2, 3, 1).detach().to("cpu", dtype=torch.float32).numpy()
                doppler_map_np = decoded_to_doppler_map(decoded_doppler_map[0], doppler_min=self.doppler_min, doppler_max=self.doppler_max)
                doppler_img_pil, _ = doppler_map_to_image(doppler_map_np, doppler_min=self.doppler_min, doppler_max=self.doppler_max)

            samples.append(RadarGenOutput(
                pd_map_np=pd_map_np, 
                pd_img_pil=pd_img_pil,
                rcs_map_np=rcs_map_np,
                rcs_img_pil=rcs_img_pil,
                doppler_map_np=doppler_map_np,
                doppler_img_pil=doppler_img_pil,
            ))

        return samples
