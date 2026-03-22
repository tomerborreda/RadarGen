# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# This file is modified from https://github.com/NVlabs/Sana/blob/main/train_scripts/train.py

import datetime
import os
import os.path as osp
import random
import time
import types
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyrallis
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from termcolor import colored

warnings.filterwarnings("ignore")  # ignore warning

from einops import rearrange

from diffusion import Scheduler
from diffusion.data.builder import build_dataloader, build_dataset
from diffusion.data.wids import DistributedRangedSampler
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_encode
from diffusion.model.respace import compute_density_for_timestep_sampling
from diffusion.model_modification import inflate_sana_input_channels_for_radargen
from diffusion.utils.checkpoint import load_checkpoint, save_checkpoint
from diffusion.utils.config import SanaConfig
from diffusion.utils.dist_utils import clip_grad_norm_, flush, get_world_size
from diffusion.utils.logger import LogBuffer, get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import DebugUnderflowOverflow, init_random_seed, set_random_seed
from diffusion.utils.optimizer import auto_scale_lr, build_optimizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
    os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "SanaBlock"


def train(config, args, accelerator, model, optimizer, lr_scheduler, train_dataloader, train_diffusion, logger):
    if getattr(config.train, "debug_nan", False):
        DebugUnderflowOverflow(model)
        logger.info("NaN debugger registered. Start to detect overflow during training.")
    log_buffer = LogBuffer()

    global_step = start_step + 1
    skip_step = max(config.train.skip_step, global_step) % train_dataloader_len
    skip_step = skip_step if skip_step < (train_dataloader_len - 20) else 0
    loss_nan_timer = 0

    # Now you train the model
    logger.info(f"Starting training loop")
    for epoch in range(start_epoch + 1, config.train.num_epochs + 1):
        time_start, last_tic = time.time(), time.time()
        sampler = (
            train_dataloader.batch_sampler.sampler if num_replicas > 1 else train_dataloader.sampler
        )
        sampler.set_epoch(epoch)
        sampler.set_start(max((skip_step - 1) * config.train.train_batch_size, 0))
        if skip_step > 1 and accelerator.is_main_process:
            logger.info(f"Skipped Steps: {skip_step}")
        skip_step = 1
        data_time_start = time.time()
        data_time_all = 0
        lm_time_all = 0
        vae_time_all = 0
        model_time_all = 0
        for step, batch in enumerate(train_dataloader):
            accelerator.wait_for_everyone()
            data_time_all += time.time() - data_time_start
            vae_time_start = time.time()

            # Extract tensors from batch using named keys (TrainingBatch format)
            pd_map = batch["pd_map"]
            rcs_map = batch["rcs_map"]
            doppler_map = batch["doppler_map"]
            bev_color_map = batch["bev_color_map"]
            bev_seg_map = batch["bev_seg_map"]
            bev_vel_map = batch["bev_velocity_map"]
            data_info = batch["data_info"]

            if load_vae_feat:
                raise NotImplementedError("VAE features loading not supported with new batch format")
            else:
                with torch.no_grad():
                    with torch.amp.autocast(
                        "cuda",
                        enabled=(config.model.mixed_precision == "fp16" or config.model.mixed_precision == "bf16"),
                    ):
                        z_pd_map = vae_encode(
                            config.vae.vae_type, vae, pd_map, config.vae.sample_posterior, accelerator.device
                        )
                        z_rcs_map = vae_encode(
                            config.vae.vae_type, vae, rcs_map, config.vae.sample_posterior, accelerator.device
                        )
                        z_doppler_map = vae_encode(
                            config.vae.vae_type, vae, doppler_map, config.vae.sample_posterior, accelerator.device
                        )
                        z_bev_color_map = vae_encode(
                            config.vae.vae_type, vae, bev_color_map, config.vae.sample_posterior, accelerator.device
                        )
                        z_bev_seg_map = vae_encode(
                            config.vae.vae_type, vae, bev_seg_map, config.vae.sample_posterior, accelerator.device
                        )
                        z_bev_vel_map = vae_encode(
                            config.vae.vae_type, vae, bev_vel_map, config.vae.sample_posterior, accelerator.device
                        )

            # Condition dropout
            if random.random() < 0.1:
                z_bev_color_map = z_bev_color_map * 0.0
            if random.random() < 0.1:
                z_bev_seg_map = z_bev_seg_map * 0.0
            if random.random() < 0.1:
                z_bev_vel_map = z_bev_vel_map * 0.0

            accelerator.wait_for_everyone()
            vae_time_all += time.time() - vae_time_start

            # Concatenate in new dim to get [B, V, C, H, W] where V = 3 here
            clean_z = torch.stack([z_pd_map, z_rcs_map, z_doppler_map], dim=1)
            clean_z = rearrange(clean_z, 'b v c h w -> (b v) c h w')  # Reshape to [B * V, C, H, W]

            lm_time_start = time.time()
            prompt = [""] * clean_z.shape[0]
            if load_text_feat:
                raise ValueError("load_text_feat should not be used.")
            else:
                if "T5" in config.text_encoder.text_encoder_name:
                    with torch.no_grad():
                        txt_tokens = tokenizer(
                            prompt, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
                        ).to(accelerator.device)
                        y = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None]
                        y_mask = txt_tokens.attention_mask[:, None, None]
                elif (
                    "gemma" in config.text_encoder.text_encoder_name or "Qwen" in config.text_encoder.text_encoder_name
                ):
                    with torch.no_grad():
                        max_length_all = config.text_encoder.model_max_length
                        txt_tokens = tokenizer(
                            prompt,
                            padding="max_length",
                            max_length=max_length_all,
                            truncation=True,
                            return_tensors="pt",
                        ).to(accelerator.device)
                        select_index = [0] + list(
                            range(-config.text_encoder.model_max_length + 1, 0)
                        )  # first bos and end N-1
                        y = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None][
                            :, :, select_index
                        ]
                        y_mask = txt_tokens.attention_mask[:, None, None][:, :, :, select_index]
                else:
                    print("error")
                    exit()

            # Sample a random timestep for each image
            bs = clean_z.shape[0]
            timesteps = torch.randint(
                0, config.scheduler.train_sampling_steps, (bs,), device=clean_z.device
            ).long()
            if config.scheduler.weighting_scheme in ["logit_normal"]:
                # adapting from diffusers.training_utils
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=config.scheduler.weighting_scheme,
                    batch_size=bs,
                    logit_mean=config.scheduler.logit_mean,
                    logit_std=config.scheduler.logit_std,
                    mode_scale=None,  # not used
                )
                timesteps = (u * config.scheduler.train_sampling_steps).long().to(clean_z.device)
            grad_norm = None
            accelerator.wait_for_everyone()
            lm_time_all += time.time() - lm_time_start
            model_time_start = time.time()
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                loss_term = train_diffusion.training_losses(
                    model, clean_z, timesteps, model_kwargs=dict(y=y, mask=y_mask, data_info=data_info, bev_condition_maps=[z_bev_color_map, z_bev_seg_map, z_bev_vel_map])
                )
                loss = loss_term["loss"].mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
                optimizer.step()
                lr_scheduler.step()
                accelerator.wait_for_everyone()
                model_time_all += time.time() - model_time_start

            if torch.any(torch.isnan(loss)):
                loss_nan_timer += 1
            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (step + 1) % config.train.log_interval == 0 or (step + 1) == 1:
                accelerator.wait_for_everyone()
                t = (time.time() - last_tic) / config.train.log_interval
                t_d = data_time_all / config.train.log_interval
                t_m = model_time_all / config.train.log_interval
                t_lm = lm_time_all / config.train.log_interval
                t_vae = vae_time_all / config.train.log_interval
                avg_time = (time.time() - time_start) / (step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                eta_epoch = str(
                    datetime.timedelta(
                        seconds=int(
                            avg_time
                            * (train_dataloader_len - sampler.step_start // config.train.train_batch_size - step - 1)
                        )
                    )
                )
                log_buffer.average()

                current_step = (
                    global_step - sampler.step_start // config.train.train_batch_size
                ) % train_dataloader_len
                current_step = train_dataloader_len if current_step == 0 else current_step
                info = (
                    f"Epoch: {epoch} | Global Step: {global_step} | Local Step: {current_step} // {train_dataloader_len}, "
                    f"total_eta: {eta}, epoch_eta:{eta_epoch}, time: all:{t:.3f}, model:{t_m:.3f}, data:{t_d:.3f}, "
                    f"lm:{t_lm:.3f}, vae:{t_vae:.3f}, lr:{lr:.3e}, "
                )
                info += (
                    f"s:({model.module.h}, {model.module.w}), "
                    if hasattr(model, "module")
                    else f"s:({model.h}, {model.w}), "
                )
                logs.update(epoch=epoch)
                logs.update(global_step=global_step)
                logs.update(local_step=current_step)
                logs.update(local_step_max=train_dataloader_len)
                logs.update(eta=eta)
                logs.update(eta_epoch=eta_epoch)

                info += ", ".join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
                model_time_all = 0
                lm_time_all = 0
                vae_time_all = 0
                if accelerator.is_main_process:
                    logger.info(info)

            logs.update(lr=lr)
            if accelerator.is_main_process:
                accelerator.log(logs, step=global_step)

            global_step += 1

            if loss_nan_timer > 20:
                raise ValueError("Loss is NaN too much times. Break here.")
            if (
                global_step % config.train.save_model_steps == 0
                or (time.time() - training_start_time) / 3600 > config.train.training_hours
            ):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    ckpt_saved_path = save_checkpoint(
                        osp.join(config.work_dir, "checkpoints"),
                        epoch=epoch,
                        step=global_step,
                        model=accelerator.unwrap_model(model),
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        generator=generator,
                        add_symlink=True,
                    )
                    if config.train.online_metric and global_step % config.train.eval_metric_step == 0 and step > 1:
                        online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                        os.makedirs(online_metric_monitor_dir, exist_ok=True)
                        with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                            f.write(osp.join(config.work_dir, "config.py") + "\n")
                            f.write(ckpt_saved_path)

                if (time.time() - training_start_time) / 3600 > config.train.training_hours:
                    logger.info(f"Stopping training at epoch {epoch}, step {global_step} due to time limit.")
                    return

            data_time_start = time.time()

        if epoch % config.train.save_model_epochs == 0 or epoch == config.train.num_epochs and not config.debug:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                ckpt_saved_path = save_checkpoint(
                    osp.join(config.work_dir, "checkpoints"),
                    epoch=epoch,
                    step=global_step,
                    model=accelerator.unwrap_model(model),
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    generator=generator,
                    add_symlink=True,
                )

                online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                os.makedirs(online_metric_monitor_dir, exist_ok=True)
                with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                    f.write(osp.join(config.work_dir, "config.py") + "\n")
                    f.write(ckpt_saved_path)
        accelerator.wait_for_everyone()


@pyrallis.wrap()
def main(cfg: SanaConfig) -> None:
    global train_dataloader_len, start_epoch, start_step, vae, generator, num_replicas, rank, training_start_time
    global load_vae_feat, load_text_feat, validation_noise, text_encoder, tokenizer
    global max_length, latent_size, null_embed_path
    global image_size, total_steps

    config = cfg
    args = cfg

    num_maps = config.model.extra['num_maps']
    num_conditions = config.model.extra['num_conditions']

    training_start_time = time.time()
    load_from = True
    if args.resume_from or config.model.resume_from:
        load_from = False
        config.model.resume_from = dict(
            checkpoint=args.resume_from or config.model.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True,
        )

    # In the sh scripts this flag is used. This modfies the batch size to a max value of 64.
    if args.debug:
        config.train.log_interval = 1
        config.train.train_batch_size = min(64, config.train.train_batch_size)
        args.report_to = "tensorboard"

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.train.use_fsdp:
        init_train = "FSDP"
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig

        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        )
    else:
        init_train = "DDP"
        fsdp_plugin = None

    accelerator = Accelerator(
        mixed_precision=config.model.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=osp.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=[init_handler],
    )

    log_name = "train_log.log"
    logger = get_root_logger(osp.join(config.work_dir, log_name))
    logger.info(accelerator.state)

    config.train.seed = init_random_seed(getattr(config.train, "seed", None))
    set_random_seed(config.train.seed + int(os.environ["LOCAL_RANK"]))
    generator = torch.Generator(device="cpu").manual_seed(config.train.seed)

    if accelerator.is_main_process:
        pyrallis.dump(config, open(osp.join(config.work_dir, "config.yaml"), "w"), sort_keys=False, indent=4)
        if args.report_to == "wandb":
            import wandb

            wandb.init(project=args.tracker_project_name, name=args.name, resume="allow", id=args.name)

    logger.info(f"Config: \n{config}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.train.seed}")
    logger.info(f"Initializing: {init_train} for training")
    image_size = config.model.image_size
    latent_size = int(image_size) // config.vae.vae_downsample_rate
    pred_sigma = getattr(config.scheduler, "pred_sigma", True)
    learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma
    max_length = config.text_encoder.model_max_length
    vae = None
    validation_noise = (
        torch.randn(1 * num_maps, config.vae.vae_latent_dim, latent_size, latent_size, device="cpu", generator=generator)
        if getattr(config.train, "deterministic_validation", False)
        else None
    )
    if not config.data.load_vae_feat:
        vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, accelerator.device).to(torch.float16)
    tokenizer = text_encoder = None
    if not config.data.load_text_feat:
        tokenizer, text_encoder = get_tokenizer_and_text_encoder(
            name=config.text_encoder.text_encoder_name, device=accelerator.device
        )
        text_embed_dim = text_encoder.config.hidden_size
    else:
        text_embed_dim = config.text_encoder.caption_channels

    logger.info(f"vae type: {config.vae.vae_type}")
    if config.text_encoder.chi_prompt:
        chi_prompt = "\n".join(config.text_encoder.chi_prompt)
        logger.info(f"Complex Human Instruct: {chi_prompt}")

    os.makedirs(config.train.null_embed_root, exist_ok=True)
    null_embed_path = osp.join(
        config.train.null_embed_root,
        f"null_embed_diffusers_{config.text_encoder.text_encoder_name}_{max_length}token_{text_embed_dim}.pth",
    )
    
    # Create the null embed if it doesn't exist. Used for the model to be able to generate images without a prompt.
    if accelerator.is_main_process and not osp.exists(null_embed_path):
        if config.data.load_text_feat and (tokenizer is None or text_encoder is None):
            logger.info(f"Loading text encoder and tokenizer from {config.text_encoder.text_encoder_name} ...")
            tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder.text_encoder_name)
    
        null_tokens = tokenizer("", max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(accelerator.device)
        if "T5" in config.text_encoder.text_encoder_name or "gemma" in config.text_encoder.text_encoder_name or "Qwen" in config.text_encoder.text_encoder_name:
            null_token_emb = text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0]
        else:
            raise ValueError(f"{config.text_encoder.text_encoder_name} is not supported!!")
        torch.save({"uncond_prompt_embeds": null_token_emb, "uncond_prompt_embeds_mask": null_tokens.attention_mask}, null_embed_path)
        if config.data.load_text_feat:
            del tokenizer
            del text_encoder
        del null_token_emb
        del null_tokens
        flush()

    os.environ["AUTOCAST_LINEAR_ATTN"] = "true" if config.model.autocast_linear_attn else "false"

    # 1. build scheduler
    train_diffusion = Scheduler(
        str(config.scheduler.train_sampling_steps),
        noise_schedule=config.scheduler.noise_schedule,
        predict_v=config.scheduler.predict_v,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=config.train.snr_loss,
        flow_shift=config.scheduler.flow_shift,
    )
    predict_info = f"v-prediction: {config.scheduler.predict_v}, noise schedule: {config.scheduler.noise_schedule}"
    if "flow" in config.scheduler.noise_schedule:
        predict_info += f", flow shift: {config.scheduler.flow_shift}"
    if config.scheduler.weighting_scheme in ["logit_normal", "mode"]:
        predict_info += (
            f", flow weighting: {config.scheduler.weighting_scheme}, "
            f"logit-mean: {config.scheduler.logit_mean}, logit-std: {config.scheduler.logit_std}"
        )
    logger.info(predict_info)

    # 2. build models
    model_kwargs = {
        "pe_interpolation": config.model.pe_interpolation,
        "config": config,
        "model_max_length": max_length,
        "qk_norm": config.model.qk_norm,
        "micro_condition": config.model.micro_condition,
        "caption_channels": text_embed_dim,
        "y_norm": config.text_encoder.y_norm,
        "attn_type": config.model.attn_type,
        "ffn_type": config.model.ffn_type,
        "mlp_ratio": config.model.mlp_ratio,
        "mlp_acts": list(config.model.mlp_acts),
        "in_channels": config.vae.vae_latent_dim,
        "y_norm_scale_factor": config.text_encoder.y_norm_scale_factor,
        "use_pe": config.model.use_pe,
        "linear_head_dim": config.model.linear_head_dim,
        "pred_sigma": pred_sigma,
        "learn_sigma": learn_sigma,
        "num_maps": num_maps,
        "num_conditions": num_conditions,
    }
    model = build_model(
        config.model.model,
        config.train.grad_checkpointing,
        getattr(config.model, "fp32_attention", False),
        input_size=latent_size,
        **model_kwargs,
    ).train()
    logger.info(
        colored(
            f"{model.__class__.__name__}:{config.model.model}, "
            f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
            "green",
            attrs=["bold"],
        )
    )
    # 2-1. load model
    if args.load_from is not None:
        config.model.load_from = args.load_from
    if config.model.load_from is not None and load_from:
        _, missing, unexpected, _ = load_checkpoint(
            config.model.load_from,
            model,
            load_ema=config.model.resume_from.get("load_ema", False) if config.model.resume_from else False,
            null_embed_path=null_embed_path,
        )
        logger.warning(f"Missing keys: {missing}")
        logger.warning(f"Unexpected keys: {unexpected}")

    
    # Model is already loaded so this can be done now. Should be before other 
    # operations and specifically before resume_from or the optimizer.
    # Duplicates the input channels in the model.
    inflate_sana_input_channels_for_radargen(model, logger)


    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # 3. build dataloader
    # Expand dataset_dir path (handle ~ and relative paths)
    if not config.data.dataset_dir.startswith(("https://", "http://", "gs://", "/", "~")):
        config.data.dataset_dir = osp.abspath(osp.expanduser(config.data.dataset_dir))
    num_replicas = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    dataset = build_dataset(
        asdict(config.data),
        resolution=image_size,
        config=config,
    )
    accelerator.wait_for_everyone()
    sampler = DistributedRangedSampler(dataset, num_replicas=num_replicas, rank=rank)
    train_dataloader = build_dataloader(
        dataset,
        num_workers=config.train.num_workers,
        batch_size=config.train.train_batch_size,
        shuffle=False,
        sampler=sampler,
    )
    train_dataloader_len = len(train_dataloader)
    load_vae_feat = getattr(train_dataloader.dataset, "load_vae_feat", False)
    load_text_feat = getattr(train_dataloader.dataset, "load_text_feat", False)

    # 4. build optimizer and lr scheduler
    lr_scale_ratio = 1
    if getattr(config.train, "auto_lr", None):
        lr_scale_ratio = auto_scale_lr(
            config.train.train_batch_size * get_world_size() * config.train.gradient_accumulation_steps,
            config.train.optimizer,
            **config.train.auto_lr,
        )
    optimizer = build_optimizer(model, config.train.optimizer)
    if config.train.lr_schedule_args and config.train.lr_schedule_args.get("num_warmup_steps", None):
        config.train.lr_schedule_args["num_warmup_steps"] = (
            config.train.lr_schedule_args["num_warmup_steps"] * num_replicas
        )
    lr_scheduler = build_lr_scheduler(config.train, optimizer, train_dataloader, lr_scale_ratio)
    logger.warning(
        f"{colored(f'Basic Setting: ', 'green', attrs=['bold'])}"
        f"lr: {config.train.optimizer['lr']:.5f}, bs: {config.train.train_batch_size}, gc: {config.train.grad_checkpointing}, "
        f"gc_accum_step: {config.train.gradient_accumulation_steps}, qk norm: {config.model.qk_norm}, "
        f"fp32 attn: {config.model.fp32_attention}, attn type: {config.model.attn_type}, ffn type: {config.model.ffn_type}, "
        f"text encoder: {config.text_encoder.text_encoder_name}, precision: {config.model.mixed_precision}"
    )

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    start_step = 0
    total_steps = train_dataloader_len * config.train.num_epochs

    # Resume training
    if config.model.resume_from is not None and config.model.resume_from["checkpoint"] is not None:
        rng_state = None
        ckpt_path = osp.join(config.work_dir, "checkpoints")
        check_flag = osp.exists(ckpt_path) and len(os.listdir(ckpt_path)) != 0
        if config.model.resume_from["checkpoint"] == "latest":
            if check_flag:
                checkpoints = os.listdir(ckpt_path)
                if "latest.pth" in checkpoints and osp.exists(osp.join(ckpt_path, "latest.pth")):
                    config.model.resume_from["checkpoint"] = osp.realpath(osp.join(ckpt_path, "latest.pth"))
                else:
                    checkpoints = [i for i in checkpoints if i.startswith("epoch_")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.replace(".pth", "").split("_")[3]))
                    config.model.resume_from["checkpoint"] = osp.join(ckpt_path, checkpoints[-1])
            else:
                config.model.resume_from["checkpoint"] = config.model.load_from

        if config.model.resume_from["checkpoint"] is not None:
            _, missing, unexpected, rng_state = load_checkpoint(
                **config.model.resume_from,
                model=model,
                optimizer=optimizer if check_flag else None,
                lr_scheduler=lr_scheduler if check_flag else None,
                null_embed_path=null_embed_path,
            )

            logger.warning(f"Missing keys: {missing}")
            logger.warning(f"Unexpected keys: {unexpected}")

            path = osp.basename(config.model.resume_from["checkpoint"])
        try:
            start_epoch = int(path.replace(".pth", "").split("_")[1]) - 1
            start_step = int(path.replace(".pth", "").split("_")[3])
        except:
            pass

        # resume randomise
        if rng_state:
            logger.info("resuming randomise")
            torch.set_rng_state(rng_state["torch"])
            np.random.set_state(rng_state["numpy"])
            random.setstate(rng_state["python"])
            generator.set_state(rng_state["generator"])  # resume generator status
            try:
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
            except:
                logger.warning("Failed to resume torch_cuda rng state")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model = accelerator.prepare(model)
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    # Start Training
    train(
        config=config,
        args=args,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        train_diffusion=train_diffusion,
        logger=logger,
    )


if __name__ == "__main__":
    main()
