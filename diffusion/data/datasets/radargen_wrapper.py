from diffusion.data.builder import DATASETS
from radargen.training.radargen_dataset import build_radargen_dataset_from_config


@DATASETS.register_module()
class RadarGenDatasetWrapper:
    """
    Wrapper to register RadarGenDataset with the diffusion builder system.

    This allows using RadarGenDataset with existing training configs by
    setting type: RadarGenDatasetWrapper in the YAML.
    
    Requires config=config to be passed to build_dataset() in train.py.
    """

    def __new__(cls, transform=None, resolution=512, config=None, **kwargs):
        """Build and return RadarGenDataset instance."""
        return build_radargen_dataset_from_config(
            config, transform=transform, resolution=resolution, **kwargs
        )
