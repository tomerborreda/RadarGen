"""Evaluation configuration using dataclasses and YAML."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    """Configuration for a single model to evaluate."""
    name: str
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None
    device: str = "cuda"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """Top-level evaluation configuration."""
    dataset_name: str
    dataset_dir: str
    dataset_version: str = "v1.0-trainval"
    eval_split: str = "val"
    illuminated_scenes_only: bool = False
    models: List[ModelConfig] = field(default_factory=list)
    compute_per_box_metrics: bool = True
    mmd_report_classes: List[str] = field(
        default_factory=lambda: ["vehicle.trailer", "vehicle.truck", "vehicle.car"]
    )
    keyframes_only: bool = True
    max_samples: Optional[int] = None
    output_dir: str = "evaluation_results"
    save_results: bool = True
    verbose: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "EvaluationConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        models = [ModelConfig(**m) for m in raw.pop("models", [])]
        return cls(models=models, **raw)
