"""Main evaluation orchestrator.

``Evaluator`` ties together the adapter, model wrappers, metrics, and
aggregation into a single ``evaluate()`` call driven by a YAML config.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from evaluation.aggregation import (
    aggregate_metrics,
    create_aggregator,
    finalize_global_metrics,
)
from evaluation.config import EvaluationConfig
from evaluation.metrics import MMDLoss, RBF, compute_all_metrics
from evaluation.models.registry import get_model
from evaluation.protocols import ModelWrapper
from radargen.core.protocols import DatasetAdapter
from radargen.datasets.registry import get_adapter
from radargen.radar_maps import filter_by_range

logger = logging.getLogger(__name__)


class Evaluator:
    """Dataset-agnostic, multi-model evaluation pipeline.

    Usage::

        config = EvaluationConfig.from_yaml("evaluation/configs/truckscenes_eval.yaml")
        evaluator = Evaluator(config)
        results = evaluator.evaluate()
        evaluator.print_summary(results)
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- load adapter ---
        self.adapter = self._create_adapter()

        # --- load models ---
        self.models: List[ModelWrapper] = []
        # Import models subpackage to trigger registration
        import evaluation.models  # noqa: F401
        for mc in config.models:
            model_wrapper = get_model(mc.name, mc, self.adapter)
            self.models.append(model_wrapper)
            logger.info(f"Loaded model: {model_wrapper.name}")

    # ------------------------------------------------------------------
    # Adapter creation
    # ------------------------------------------------------------------

    def _create_adapter(self) -> DatasetAdapter:
        """Instantiate the dataset adapter based on config."""
        name = self.config.dataset_name

        if name == "truckscenes":
            from truckscenes import TruckScenes

            trucksc = TruckScenes(
                self.config.dataset_version,
                self.config.dataset_dir,
                True,
            )
            return get_adapter(
                "truckscenes",
                trucksc=trucksc,
                illuminated_only=self.config.illuminated_scenes_only,
            )

        if name == "nuscenes":
            from nuscenes.nuscenes import NuScenes

            nusc = NuScenes(
                version=self.config.dataset_version,
                dataroot=self.config.dataset_dir,
                verbose=True,
            )
            return get_adapter("nuscenes", nusc=nusc)

        # Generic fallback
        return get_adapter(name, data_root=self.config.dataset_dir, version=self.config.dataset_version)

    # ------------------------------------------------------------------
    # Ground-truth PCL extraction
    # ------------------------------------------------------------------

    def _get_gt_pcl(self, sample_data: dict) -> np.ndarray:
        """Load, transform, and filter the ground-truth radar PCL.

        Returns (N, 4) array [x, y, rcs, doppler] in ego flat-up frame.
        """
        ref_token = self.adapter.get_reference_sensor_token(sample_data)
        ref_to_flat_up = self.adapter.get_ref_to_vehicle_flat_up_transform(ref_token)

        # load_radar_pointcloud already returns points in the flat-up frame
        pcl = self.adapter.load_radar_pointcloud(sample_data, nsweeps=1)
        if pcl.shape[0] == 0:
            raise RuntimeError(f"No points found in radar sensor for sample {sample_data}.")

        # Filter by camera frustum (points are in flat-up frame)
        pcl_filtered = self.adapter.filter_by_camera_frustum(
            pcl,
            sample_data,
            ref_to_points_transform=ref_to_flat_up,
        )

        # Filter by square coordinate range
        coordinate_range = self.adapter.normalization.coordinate_range
        pcl_filtered = filter_by_range(pcl_filtered, coordinate_range)

        if pcl_filtered.shape[0] == 0:
            raise RuntimeError(f"No points left after filtering for sample {sample_data}.")

        # columns: [x, y, z, vx, vy, vz, rcs, doppler] → [x, y, rcs, doppler]
        return pcl_filtered[:, [0, 1, 6, 7]]

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, dict]:
        """Run evaluation for all models and return per-model final metrics."""
        norm_config = self.adapter.normalization
        mmd_report_classes = self.config.mmd_report_classes

        # One aggregator per model
        aggregators: Dict[str, dict] = {}
        for mw in self.models:
            aggregators[mw.name] = create_aggregator()

        mmd_loss = MMDLoss(kernel=RBF(device=self.device), device=self.device)

        # Gather samples
        all_samples = self._gather_samples()
        logger.info(f"Total samples to evaluate: {len(all_samples)}")

        with torch.no_grad():
            for idx, (sample_data, sample_data_next, bounding_boxes, scene_token, frame_idx) in tqdm(
                enumerate(all_samples),
                total=len(all_samples),
                desc="Evaluating",
            ):
                gt_pcl = self._get_gt_pcl(sample_data)

                for mw in self.models:
                    logger.info(f"  [{idx+1}/{len(all_samples)}] {mw.name}")
                    syn_pcl = mw.predict_point_cloud(
                        self.adapter, sample_data, sample_data_next, scene_token, frame_idx
                    )

                    metrics = compute_all_metrics(
                        syn_pcl, gt_pcl, bounding_boxes, norm_config, mmd_loss, self.device,
                    )
                    aggregate_metrics(aggregators[mw.name], metrics, mmd_report_classes)

        # Finalise
        results: Dict[str, dict] = {}
        for mw in self.models:
            logger.info(f"Finalizing metrics for {mw.name}...")
            results[mw.name] = finalize_global_metrics(
                aggregators[mw.name], self.device, mmd_report_classes, logger,
            )

        if self.config.save_results:
            self.save_results(results)

        return results

    # ------------------------------------------------------------------
    # Sample gathering
    # ------------------------------------------------------------------

    @staticmethod
    def _as_sample_data(sample: Any) -> dict:
        """Normalize a sample object to flat sample_data format."""
        return sample["data"] if isinstance(sample, dict) and "data" in sample else sample

    def _gather_samples(self):
        """Collect (sample_data, sample_data_next, bounding_boxes, scene_token, frame_idx) tuples."""
        split = self.config.eval_split
        samples = []

        if self.config.keyframes_only:
            for scene_token, indexed_keyframes in self.adapter.iter_keyframes_indexed(split):
                for sample, frame_idx in indexed_keyframes:
                    sample_data = self._as_sample_data(sample)
                    sample_data_next = self.adapter.get_next_immediate_sample(sample_data)
                    if sample_data_next is None:
                        raise RuntimeError(
                            f"Sample has no immediate next frame: {sample_data}, frame_idx={frame_idx}"
                        )

                    if self.config.compute_per_box_metrics:
                        ref_token = self.adapter.get_reference_sensor_token(sample_data)
                        bounding_boxes = self.adapter.get_bounding_boxes(sample_data, ref_sensor_token=ref_token)
                    else:
                        bounding_boxes = []

                    samples.append((sample_data, sample_data_next, bounding_boxes, scene_token, frame_idx))

                    if self.config.max_samples and len(samples) >= self.config.max_samples:
                        return samples
        else:
            for scene_token, scene_samples in self.adapter.iter_scenes(split):
                for frame_idx in range(len(scene_samples) - 1):
                    sample_data = self._as_sample_data(scene_samples[frame_idx])
                    sample_data_next = self._as_sample_data(scene_samples[frame_idx + 1])

                    if self.config.compute_per_box_metrics:
                        ref_token = self.adapter.get_reference_sensor_token(sample_data)
                        bounding_boxes = self.adapter.get_bounding_boxes(sample_data, ref_sensor_token=ref_token)
                    else:
                        bounding_boxes = []

                    samples.append((sample_data, sample_data_next, bounding_boxes, scene_token, frame_idx))

                    if self.config.max_samples and len(samples) >= self.config.max_samples:
                        return samples

        return samples

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_results(self, results: Dict[str, dict]) -> str:
        """Save results as JSON. Returns path to the output file."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.config.output_dir, f"eval_results_{ts}.json")

        # Convert numpy types for JSON serialization
        serializable = {}
        for model_name, metrics in results.items():
            serializable[model_name] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in metrics.items()
            }

        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)
        logger.info(f"Results saved to {path}")
        return path

    def print_summary(self, results: Dict[str, dict]) -> None:
        """Log a formatted comparison of all models."""
        logger.info("=" * 60)
        logger.info("EVALUATION RESULTS SUMMARY")
        logger.info("=" * 60)

        for model_name, metrics in results.items():
            logger.info(f"\n--- {model_name} ---")
            for key in sorted(metrics.keys()):
                val = metrics[key]
                if isinstance(val, float):
                    logger.info(f"  {key}: {val:.6f}")
                else:
                    logger.info(f"  {key}: {val}")
