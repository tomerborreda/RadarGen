"""
RadarGen: Multi-dataset radar point cloud generation from camera images.

This package provides a clean, dataset-agnostic interface for:
- Training diffusion models that generate radar maps from BEV conditioning
- Running inference to generate radar point clouds from camera images
- Evaluating generated radar data against ground truth
"""

from radargen.core.data_types import RadarGenInferenceInput, TrainingBatch
from radargen.core.protocols import DatasetAdapter, DatasetConfig
from radargen.datasets.registry import get_adapter, register_adapter

__all__ = [
    "RadarGenInferenceInput",
    "TrainingBatch",
    "DatasetAdapter",
    "DatasetConfig",
    "get_adapter",
    "register_adapter",
]
