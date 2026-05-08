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
    """Two-phase pack-then-balance.

    Phase 1 (online): use best-fit to fill ``balance_window`` packs.
    Phase 2 (batched): run swap-based local search to minimize the
    length variance across those packs.
    Phase 3: yield all balanced packs in one batch.

    Trades a small amount of latency (one window's worth of samples) for
    significantly more uniform pack lengths -- which directly translates
    to better DP utilization since each step's wall time is bounded by
    the slowest rank's token count.

    Note: pack ordering is not preserved within a window. Samples within a
    window may be reshuffled across packs by the balancing step.
    """

    def __init__(
        self,
        packing_length: int,
        balance_window: int = 64,
        num_open_buckets: int = 8,
        max_swap_iters: int = 200,
        min_gain: int = 1,
    ) -> None:
        super().__init__(packing_length)
        if balance_window < 1:
            raise ValueError("balance_window must be >= 1")
        if num_open_buckets < 1:
            raise ValueError("num_open_buckets must be >= 1")
        self.balance_window = balance_window
        self.num_open_buckets = num_open_buckets
        self.max_swap_iters = max_swap_iters
        self.min_gain = min_gain

        # Open buckets being filled. Each bucket = [length, [(item, length), ...]].
        # We carry per-item lengths because the balance phase needs them.
        self._open: List[List[Any]] = []
        # Completed packs awaiting balancing. Same shape as buckets.
        self._completed: List[List[Any]] = []

    # ---- Phase 1: best-fit fill -----------------------------------------
    def add(self, item: Any, length: int) -> Iterator[List[Any]]:
        n = self.packing_length

        # Find best-fit open bucket.
        best_idx = -1
        best_remain = n + 1
        for i, (blen, _) in enumerate(self._open):
            remain = n - blen - length
            if 0 <= remain < best_remain:
                best_remain = remain
                best_idx = i

        if best_idx == -1:
            if len(self._open) < self.num_open_buckets:
                self._open.append([length, [(item, length)]])
            else:
                # All slots taken; evict the fullest into _completed.
                evict_idx = max(range(len(self._open)), key=lambda i: self._open[i][0])
                self._completed.append(self._open[evict_idx])
                self._open[evict_idx] = [length, [(item, length)]]
        else:
            self._open[best_idx][0] += length
            self._open[best_idx][1].append((item, length))

        if len(self._completed) >= self.balance_window:
            yield from self._balance_and_emit()

    def flush(self) -> Iterator[List[Any]]:
        # Move all still-open buckets into completed, then balance & emit.
        for blen, pairs in self._open:
            if pairs:
                self._completed.append([blen, pairs])
        self._open = []
        if self._completed:
            yield from self._balance_and_emit()

    # ---- Phase 2 + 3: balance and yield ---------------------------------
    def _balance_and_emit(self) -> Iterator[List[Any]]:
        sums: List[int] = [c[0] for c in self._completed]
        packs: List[List[Tuple[Any, int]]] = [c[1] for c in self._completed]
        self._completed = []

        if len(packs) >= 2:
            self._balance(packs, sums)

        for pairs in packs:
            if pairs:
                yield [item for item, _ in pairs]

    def _balance(
        self,
        packs: List[List[Tuple[Any, int]]],
        sums: List[int],
    ) -> None:
        """Swap-based local search to reduce length variance.

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
