"""Online (streaming) packing strategies."""

from typing import Any, Iterator, List, Tuple

from .base import OnlinePackingStrategy


class NextFitPacking(OnlinePackingStrategy):
    """Single-buffer next-fit packing.

    Behaviorally equivalent to the original logic in
    ``MultiModalIterableDataset.__iter__``: maintain one open buffer; whenever
    the next sample doesn't fit, flush the buffer and start a new one with
    that sample.

    This is the simplest online strategy and serves as a baseline. It does not
    look ahead and tends to leave tail space unused.
    """

    def __init__(self, packing_length: int) -> None:
        super().__init__(packing_length)
        self._buffer: List[Any] = []
        self._buffer_length: int = 0

    def add(self, item: Any, length: int) -> Iterator[List[Any]]:
        # If adding this sample would overflow, flush the current buffer first.
        if self._buffer_length > 0 and self._buffer_length + length > self.packing_length:
            yield self._buffer
            self._buffer = []
            self._buffer_length = 0

        self._buffer.append(item)
        self._buffer_length += length

    def flush(self) -> Iterator[List[Any]]:
        if self._buffer:
            yield self._buffer
        self._buffer = []
        self._buffer_length = 0


class BestFitPacking(OnlinePackingStrategy):
    """K-bucket best-fit packing.

    Maintains up to ``num_open_buckets`` open packs. Each incoming sample is
    placed into the bucket whose remaining capacity is the smallest one that
    can still accommodate it. A bucket is emitted when it crosses
    ``fill_threshold * packing_length``; if no bucket can fit the sample and
    we already have ``num_open_buckets`` open, the fullest bucket is evicted
    to make room.

    This significantly reduces tail waste compared to next-fit by allowing
    multiple in-flight packs at once.
    """

    def __init__(
        self,
        packing_length: int,
        num_open_buckets: int = 4,
        fill_threshold: float = 0.95,
    ) -> None:
        super().__init__(packing_length)
        if num_open_buckets < 1:
            raise ValueError("num_open_buckets must be >= 1")
        if not 0.0 < fill_threshold <= 1.0:
            raise ValueError("fill_threshold must be in (0, 1]")
        self.num_open_buckets = num_open_buckets
        self.fill_full = int(packing_length * fill_threshold)
        # Each bucket is [length, items].
        self._buckets: List[List[Any]] = []

    def _find_best_bucket(self, length: int) -> int:
        """Return the index of the bucket with the smallest remaining space
        that can still fit ``length``, or -1 if none can."""
        best_idx = -1
        best_remain = self.packing_length + 1
        for i, (blen, _) in enumerate(self._buckets):
            remain = self.packing_length - blen - length
            if 0 <= remain < best_remain:
                best_remain = remain
                best_idx = i
        return best_idx

    def add(self, item: Any, length: int) -> Iterator[List[Any]]:
        idx = self._find_best_bucket(length)

        if idx == -1:
            # Nothing can fit this sample.
            if len(self._buckets) < self.num_open_buckets:
                self._buckets.append([length, [item]])
            else:
                # Evict the fullest bucket to make room.
                evict_idx = max(range(len(self._buckets)), key=lambda i: self._buckets[i][0])
                yield self._buckets[evict_idx][1]
                self._buckets[evict_idx] = [length, [item]]
            return

        self._buckets[idx][0] += length
        self._buckets[idx][1].append(item)

        # Emit early if the bucket is "full enough" -- no point keeping it open.
        if self._buckets[idx][0] >= self.fill_full:
            yield self._buckets.pop(idx)[1]

    def flush(self) -> Iterator[List[Any]]:
        for blen, items in self._buckets:
            if items:
                yield items
        self._buckets = []


class BalancedPacking(OnlinePackingStrategy):
    """Streaming pack-then-balance with a single ``num_buckets``-sized buffer.

    Maintains exactly one list of up to ``num_buckets`` open packs. Each
    incoming sample is placed via best-fit. When no bucket can accept the
    sample and all ``num_buckets`` slots are taken, the packer:

    1. runs swap-based local search across all open buckets to minimize
       length variance;
    2. yields the oldest bucket (FIFO);
    3. opens a new bucket containing the current sample in the freed slot.

    This produces a smooth 1-in-1-out yield rhythm at steady state, so the
    DataLoader's worker prefetch can keep the pipeline full -- avoiding
    the periodic stall that batch-balanced strategies suffer from when
    they ``yield`` an entire window at once.

    Pack ordering reflects bucket creation order (not original sample
    order), since balancing reshuffles items across the open buckets.
    """

    def __init__(
        self,
        packing_length: int,
        num_buckets: int = 8,
        max_swap_iters: int = 200,
        min_gain: int = 1,
    ) -> None:
        super().__init__(packing_length)
        if num_buckets < 1:
            raise ValueError("num_buckets must be >= 1")
        self.num_buckets = num_buckets
        self.max_swap_iters = max_swap_iters
        self.min_gain = min_gain

        # Each bucket = [length, [(item, item_length), ...]].
        # We carry per-item lengths because the balance step needs them.
        # List order is creation order (oldest at index 0).
        self._buckets: List[List[Any]] = []

    def _find_best_bucket(self, length: int) -> int:
        """Return index of best-fit bucket for ``length``, or -1 if none can fit."""
        best_idx = -1
        best_remain = self.packing_length + 1
        for i, (blen, _) in enumerate(self._buckets):
            remain = self.packing_length - blen - length
            if 0 <= remain < best_remain:
                best_remain = remain
                best_idx = i
        return best_idx

    def add(self, item: Any, length: int) -> Iterator[List[Any]]:
        idx = self._find_best_bucket(length)

        if idx >= 0:
            # Best-fit: append to existing bucket. No yield.
            self._buckets[idx][0] += length
            self._buckets[idx][1].append((item, length))
            return

        # Nothing can fit this sample.
        if len(self._buckets) < self.num_buckets:
            # Cold start: open a new bucket. No yield.
            self._buckets.append([length, [(item, length)]])
            return

        # All num_buckets slots are taken AND none can accept the sample.
        # Balance everything, evict the oldest, open a fresh bucket.
        sums = [b[0] for b in self._buckets]
        packs = [b[1] for b in self._buckets]
        if len(packs) >= 2:
            self._balance(packs, sums)
        # After balancing, write back so the remaining buckets see the
        # updated contents on the next add.
        for i, (s, p) in enumerate(zip(sums, packs)):
            self._buckets[i] = [s, p]

        oldest = self._buckets.pop(0)
        yield [it for it, _ in oldest[1]]

        self._buckets.append([length, [(item, length)]])

    def flush(self) -> Iterator[List[Any]]:
        if len(self._buckets) >= 2:
            sums = [b[0] for b in self._buckets]
            packs = [b[1] for b in self._buckets]
            self._balance(packs, sums)
            self._buckets = [[s, p] for s, p in zip(sums, packs)]

        for _, pairs in self._buckets:
            if pairs:
                yield [it for it, _ in pairs]
        self._buckets = []

    def _balance(
        self,
        packs: List[List[Tuple[Any, int]]],
        sums: List[int],
    ) -> None:
        """Swap-based local search to reduce length variance across packs.

        Each iteration picks the (max, min) pair and tries:
        1. A one-way move from max -> min that strictly shrinks the gap.
        2. A swap (item from max <-> item from min) that strictly shrinks
           the gap by more than ``min_gain``.
        Stops when no improvement is possible or after ``max_swap_iters``.
        """
        n = self.packing_length

        for _ in range(self.max_swap_iters):
            hi = max(range(len(sums)), key=lambda i: sums[i])
            lo = min(range(len(sums)), key=lambda i: sums[i])
            gap = sums[hi] - sums[lo]
            if gap <= self.min_gain:
                break

            # Try a one-way move first.
            best_move_idx = -1
            best_new_gap = gap
            for idx, (_, ilen) in enumerate(packs[hi]):
                if sums[lo] + ilen > n:
                    continue
                new_gap = abs((sums[hi] - ilen) - (sums[lo] + ilen))
                if new_gap < best_new_gap:
                    best_new_gap = new_gap
                    best_move_idx = idx

            if best_move_idx >= 0:
                item, ilen = packs[hi].pop(best_move_idx)
                packs[lo].append((item, ilen))
                sums[hi] -= ilen
                sums[lo] += ilen
                continue

            # Fall back to a swap.
            best_swap = None
            best_new_gap = gap
            for i, (_, a) in enumerate(packs[hi]):
                for j, (_, b) in enumerate(packs[lo]):
                    if a <= b:
                        continue
                    new_hi = sums[hi] - a + b
                    new_lo = sums[lo] - b + a
                    if new_hi > n or new_lo > n:
                        continue
                    new_gap = abs(new_hi - new_lo)
                    if new_gap + self.min_gain < gap and new_gap < best_new_gap:
                        best_new_gap = new_gap
                        best_swap = (i, j)

            if best_swap is None:
                break

            i, j = best_swap
            packs[hi][i], packs[lo][j] = packs[lo][j], packs[hi][i]
            sums[hi] = sum(l for _, l in packs[hi])
            sums[lo] = sum(l for _, l in packs[lo])
