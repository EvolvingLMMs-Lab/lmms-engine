"""Base classes for packing strategies.

Two flavors:
- OfflinePackingStrategy: all sample lengths known up-front; returns index groups.
- OnlinePackingStrategy: streaming; samples arrive one-by-one, packs are emitted
  as soon as they are ready.

Only the online side is implemented for now (RFC). Offline will be migrated
later from `naive/multimodal_dataset.py`.
"""

from abc import ABC, abstractmethod
from typing import Any, Iterator, List


class OnlinePackingStrategy(ABC):
    """Pack samples in a streaming fashion.

    Typical usage from an iterable dataset::

        packer = SomeOnlinePacker(packing_length)
        for item, length in stream:
            yield from packer.add(item, length)
        yield from packer.flush()

    Implementations are responsible for buffering and deciding when a pack is
    "ready" to be emitted. Oversized samples (length > packing_length) are
    expected to be handled by the caller before `add` is invoked.
    """

    def __init__(self, packing_length: int) -> None:
        self.packing_length = packing_length

    @abstractmethod
    def add(self, item: Any, length: int) -> Iterator[List[Any]]:
        """Add one sample.

        Args:
            item: The opaque sample object (e.g. a data dict). The packer never
                inspects it; callers consume it back via the yielded packs.
            length: The token length of `item`. Must satisfy
                ``length <= self.packing_length``.

        Yields:
            Zero or more packed batches. Each batch is a list of `item`s whose
            total length is ``<= self.packing_length``.
        """

    @abstractmethod
    def flush(self) -> Iterator[List[Any]]:
        """Yield any remaining buffered items as final packs.

        Should be called exactly once when the input stream is exhausted.
        After `flush`, the packer's internal state is reset so the instance
        can be reused for a new stream.
        """
