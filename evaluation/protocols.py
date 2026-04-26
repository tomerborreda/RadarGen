"""Abstract base class for evaluation model wrappers."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from radargen.core.protocols import DatasetAdapter


class ModelWrapper(ABC):
    """
    Abstract interface for models that produce point clouds for evaluation.

    Each implementation wraps a specific model (RadarGen, Ground Truth, baselines)
    and provides a unified interface for the Evaluator to call.

    The output is always (N, 4) with columns [x, y, rcs, doppler] in the
    ego vehicle flat-up frame, using original (unnormalized) value ranges.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name for this model in results."""
        pass

    @abstractmethod
    def predict_point_cloud(
        self,
        adapter: DatasetAdapter,
        sample_data: dict,
        sample_data_next: dict = None,
        scene_token: Optional[str] = None,
        frame_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a point cloud prediction for a single sample.

        Args:
            adapter: DatasetAdapter instance for data access.
            sample_data: Current sample's data dictionary (sensor tokens).
            sample_data_next: Optional next sample's data dictionary
                             (for models that use two consecutive timesteps).
            scene_token: Optional scene identifier for loading pre-computed data.
                        Useful for models that can use cached/pre-computed
                        intermediate representations (e.g., BEV maps).
            frame_idx: Optional frame index within scene for loading pre-computed
                      data. Should be the 0-based index matching the order in
                      adapter.iter_scenes().

        Returns:
            np.ndarray: Point cloud of shape (N, 4) with columns
                       [x, y, rcs, doppler] in ego vehicle coordinate frame,
                       using original (unnormalized) value ranges.
        """
        pass
