"""Configuration for BEV map generation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class BEVConditionConfig:
    """Configuration for generating BEV conditioning maps.

    Attributes:
        dataset_name: Dataset identifier (e.g., "truckscenes", "nuscenes")
        dataset_dir: Path to dataset root directory
        dataset_version: Dataset version string (e.g., "v1.0-trainval", "v1.0-mini")
        save_dir: Directory to save generated BEV maps
        resolution: Output map resolution in pixels (default: 512)
        camera_views: Optional list of camera views to use (None = use all from adapter)
        split: Optional dataset split to process (e.g., "train", "val")
        use_batched_inference: Batch cameras through foundation models (3-5x speedup)
        use_async_io: Enable async disk writes (10-15% speedup)
        use_torch_compile: torch.compile() optimization (1.5-2x additional speedup, requires PyTorch 2.0+)
        use_mixed_precision: Use fp16 for segmentation model (saves GPU memory)

    Note:
        coordinate_range and camera_freq are obtained from the adapter's NormalizationConfig.
    """

    dataset_name: str
    dataset_dir: Path
    dataset_version: str
    save_dir: Path
    resolution: int = 512
    camera_views: Optional[List[str]] = None
    split: Optional[str] = None
    # Optimization options
    use_batched_inference: bool = True
    use_async_io: bool = True
    use_torch_compile: bool = False
    use_mixed_precision: bool = False

    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.dataset_dir, str):
            self.dataset_dir = Path(self.dataset_dir)
        if isinstance(self.save_dir, str):
            self.save_dir = Path(self.save_dir)

        # Create save directory if it doesn't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)
