"""Registry for dataset adapters."""

from typing import Any, Callable, Dict, List, Optional, Type

from radargen.core.protocols import DatasetAdapter

# Global registry mapping dataset names to adapter classes
_ADAPTER_REGISTRY: Dict[str, Type[DatasetAdapter]] = {}


def register_adapter(name: str) -> Callable[[Type[DatasetAdapter]], Type[DatasetAdapter]]:
    """
    Decorator to register a dataset adapter class.

    Usage:
        @register_adapter("truckscenes")
        class TruckScenesAdapter(DatasetAdapter):
            ...

    Args:
        name: Unique name for the adapter (e.g., "truckscenes", "nuscenes")

    Returns:
        Decorator function that registers the class
    """
    def decorator(cls: Type[DatasetAdapter]) -> Type[DatasetAdapter]:
        if name in _ADAPTER_REGISTRY:
            raise ValueError(
                f"Adapter '{name}' is already registered. "
                f"Existing: {_ADAPTER_REGISTRY[name]}, New: {cls}"
            )
        _ADAPTER_REGISTRY[name] = cls
        return cls
    return decorator


def get_adapter(
    name: str,
    **kwargs: Any,
) -> DatasetAdapter:
    """
    Get an instance of a registered dataset adapter.

    Args:
        name: Name of the registered adapter
        **kwargs: Arguments to pass to the adapter constructor

    Returns:
        Instantiated adapter

    Raises:
        KeyError: If adapter name is not registered
    """
    if name not in _ADAPTER_REGISTRY:
        available = list(_ADAPTER_REGISTRY.keys())
        raise KeyError(
            f"Adapter '{name}' is not registered. Available adapters: {available}"
        )
    adapter_cls = _ADAPTER_REGISTRY[name]
    return adapter_cls(**kwargs)


def get_adapter_class(name: str) -> Type[DatasetAdapter]:
    """
    Get the class of a registered dataset adapter.

    Args:
        name: Name of the registered adapter

    Returns:
        Adapter class (not instantiated)

    Raises:
        KeyError: If adapter name is not registered
    """
    if name not in _ADAPTER_REGISTRY:
        available = list(_ADAPTER_REGISTRY.keys())
        raise KeyError(
            f"Adapter '{name}' is not registered. Available adapters: {available}"
        )
    return _ADAPTER_REGISTRY[name]


def list_adapters() -> List[str]:
    """
    List all registered adapter names.

    Returns:
        List of registered adapter names
    """
    return list(_ADAPTER_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """
    Check if an adapter is registered.

    Args:
        name: Name to check

    Returns:
        True if adapter is registered, False otherwise
    """
    return name in _ADAPTER_REGISTRY
