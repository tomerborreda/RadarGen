"""RadarGen model wrapper for evaluation."""

import logging
from typing import Optional

import numpy as np
import torch

from evaluation.models.registry import register_model
from evaluation.protocols import ModelWrapper
from radargen.core.protocols import DatasetAdapter
from radargen.inference.radargen_inference import RadarGenInference

logger = logging.getLogger(__name__)


@register_model("radargen")
def create_radargen(config, adapter, **kwargs):
    return RadarGenWrapper(config, adapter)


class RadarGenWrapper(ModelWrapper):
    """Evaluation wrapper around ``RadarGenInference``."""

    def __init__(self, config, adapter: DatasetAdapter):
        self._config = config
        self.adapter = adapter

        device_str = getattr(config, "device", "cuda")
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")

        params = getattr(config, "params", {}) or {}
        self.num_inference_steps = params.get("num_inference_steps", 20)
        self.seed = params.get("seed", 42)
        self.image_size = params.get("image_size", 512)

        # BEV maps directory for pre-computed maps (optional)
        self.bev_maps_dir = params.get("bev_maps_dir", None)
        if self.bev_maps_dir:
            logger.info(f"Pre-computed BEV maps will be loaded from: {self.bev_maps_dir}")
        else:
            logger.info("BEV maps will be computed from camera images (full pipeline)")

        config_path = getattr(config, "config_path", None)
        checkpoint_path = getattr(config, "checkpoint_path", None)

        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.model = RadarGenInference(
            config=config_path,
            device=self.device,
            generator=generator,
            adapter=adapter,
        )
        if checkpoint_path:
            logger.info(f"Loading RadarGen checkpoint: {checkpoint_path}")
            self.model.load_checkpoint(checkpoint_path)

        self.model.eval()

    @property
    def name(self) -> str:
        return "RadarGen"

    def predict_point_cloud(
        self,
        adapter: DatasetAdapter,
        sample_data: dict,
        sample_data_next: dict = None,
        scene_token: Optional[str] = None,
        frame_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Run RadarGen inference for a sample and return [x, y, rcs, doppler].

        If bev_maps_dir is configured and scene_token/frame_idx are provided,
        loads pre-computed BEV maps to skip stage 1 (BEV generation), significantly
        speeding up evaluation. Otherwise, runs the full 3-stage pipeline.

        Args:
            adapter: DatasetAdapter instance
            sample_data: Sample data dictionary
            sample_data_next: Next sample data (for velocity computation)
            scene_token: Scene identifier for loading pre-computed BEV maps
            frame_idx: Frame index within scene for loading pre-computed BEV maps
        """
        with torch.no_grad():
            # Check if we should use pre-computed BEV maps
            if self.bev_maps_dir and scene_token is not None and frame_idx is not None:
                try:
                    # Load pre-computed BEV maps from disk
                    bev_color, bev_seg, bev_vel = adapter.load_bev_condition_maps(
                        scene_token=scene_token,
                        frame_idx=frame_idx,
                        bev_maps_dir=self.bev_maps_dir,
                    )

                    # Run inference with pre-computed maps (skip stage 1)
                    pcl = self.model.get_pcl_from_bev_maps(
                        bev_color,
                        bev_seg,
                        bev_vel,
                        num_inference_steps=self.num_inference_steps,
                    )

                except FileNotFoundError as e:
                    logger.warning(
                        f"BEV maps not found for scene {scene_token}, frame {frame_idx}: {e}."
                    )
                    raise e
            else:
                # Run full 3-stage pipeline (BEV generation → Diffusion → Deconvolution)
                pcl = self.model.from_sample_data(
                    sample_data,
                    sample_data_next,
                    num_inference_steps=self.num_inference_steps,
                )

        if pcl is None or pcl.shape[0] == 0:
            raise ValueError("Invalid point cloud")

        # RadarGen output is (N, 5) = [x, y, z, rcs, doppler]
        # Drop z to get (N, 4) = [x, y, rcs, doppler]
        if pcl.shape[1] == 5:
            return pcl[:, [0, 1, 3, 4]]
        # Already (N, 4)
        return pcl[:, :4]
