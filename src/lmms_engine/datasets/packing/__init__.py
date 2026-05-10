"""Packing strategies for grouping variable-length samples into fixed-budget packs.

Currently implements the online (streaming) side only. Offline strategies used
by map-style datasets will be migrated here later.
"""

from .base import OnlinePackingStrategy
from .online import BalancedPacking, BestFitPacking, FirstFitPacking

_ONLINE_REGISTRY = {
    "first_fit": FirstFitPacking,
    "best_fit": BestFitPacking,
    "balanced": BalancedPacking,
}


def build_online_packer(strategy: str, packing_length: int, **kwargs) -> OnlinePackingStrategy:
    """Construct an online packer by name.

    Args:
        strategy: Strategy name. One of ``"first_fit"``, ``"best_fit"``,
            ``"balanced"``.
        packing_length: Per-pack token budget.
        **kwargs: Forwarded to the concrete strategy's constructor.
    """
    if strategy not in _ONLINE_REGISTRY:
        raise ValueError(f"Unknown online packing strategy: {strategy!r}. " f"Available: {sorted(_ONLINE_REGISTRY)}")
    return _ONLINE_REGISTRY[strategy](packing_length, **kwargs)


__all__ = [
    "OnlinePackingStrategy",
    "FirstFitPacking",
    "BestFitPacking",
    "BalancedPacking",
    "build_online_packer",
]
