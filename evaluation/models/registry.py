"""Registry for evaluation model wrappers."""

from typing import Any, Callable, Dict, List, Type

from evaluation.protocols import ModelWrapper


# Global registry mapping model names to factory functions.
# Factory signature: factory(config: ModelConfig, adapter: DatasetAdapter, **kwargs) -> ModelWrapper
_MODEL_REGISTRY: Dict[str, Callable[..., ModelWrapper]] = {}


def register_model(name: str) -> Callable:
    """
    Decorator to register a model factory function.

    The decorated callable receives (config, adapter, **kwargs) and must
    return a ``ModelWrapper`` instance.

    Usage::

        @register_model("my_model")
        def create_my_model(config, adapter, **kwargs):
            return MyModelWrapper(config, adapter)
    """
    def decorator(fn: Callable[..., ModelWrapper]) -> Callable[..., ModelWrapper]:
        if name in _MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' is already registered. "
                f"Existing: {_MODEL_REGISTRY[name]}, New: {fn}"
            )
        _MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def get_model(name: str, config: Any, adapter: Any, **kwargs: Any) -> ModelWrapper:
    """
    Instantiate a registered model wrapper.

    Args:
        name: Registered model name (e.g. "radargen", "ground_truth").
        config: ``ModelConfig`` instance for this model.
        adapter: ``DatasetAdapter`` instance.
        **kwargs: Extra keyword arguments forwarded to the factory.

    Returns:
        Instantiated ``ModelWrapper``.

    Raises:
        KeyError: If the model name is not registered.
    """
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise KeyError(
            f"Model '{name}' is not registered. Available models: {available}"
        )
    return _MODEL_REGISTRY[name](config, adapter, **kwargs)


def list_models() -> List[str]:
    """Return all registered model names."""
    return list(_MODEL_REGISTRY.keys())
