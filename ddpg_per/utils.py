"""
Utility classes used by the DDPG+PER implementation.

Classes
-------
LinearSchedule
    Linearly anneals a value from ``initial_p`` to ``final_p`` over a fixed
    number of time-steps.
SegmentTree
    Base segment-tree data structure (supports efficient prefix-sum and
    prefix-min queries in O(log n)).
SumSegmentTree
    Segment tree whose ``reduce`` operation is addition.
MinSegmentTree
    Segment tree whose ``reduce`` operation is minimum.
"""

import operator


class LinearSchedule:
    """Linearly anneal a scalar value over a fixed number of time-steps.

    After ``schedule_timesteps`` the value is clamped at ``final_p``.

    Parameters
    ----------
    schedule_timesteps : int
        Number of time-steps over which to anneal from *initial_p* to
        *final_p*.
    final_p : float
        Value returned at (and after) *schedule_timesteps*.
    initial_p : float, optional
        Value returned at time-step 0 (default: ``1.0``).

    Examples
    --------
    >>> sched = LinearSchedule(100, final_p=1.0, initial_p=0.4)
    >>> sched.value(0)
    0.4
    >>> sched.value(50)
    0.7
    >>> sched.value(100)
    1.0
    """

    def __init__(
        self,
        schedule_timesteps: int,
        final_p: float,
        initial_p: float = 1.0,
    ) -> None:
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t: int) -> float:
        """Return the scheduled value at time-step *t*.

        Parameters
        ----------
        t : int
            Current time-step.

        Returns
        -------
        float
            Linearly interpolated value between *initial_p* and *final_p*.
        """
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class SegmentTree:
    """Segment tree for efficient range-reduce queries.

    Supports O(log n) updates and O(log n) range queries.
    See https://en.wikipedia.org/wiki/Segment_tree for background.

    Parameters
    ----------
    capacity : int
        Total number of leaf elements — **must** be a positive power of two.
    operation : callable
        Binary associative operation used to combine two elements (e.g.
        ``operator.add`` or ``min``).
    neutral_element :
        Identity element for *operation* (e.g. ``0.0`` for addition,
        ``float('inf')`` for minimum).
    """

    def __init__(self, capacity: int, operation, neutral_element) -> None:
        if not (capacity > 0 and (capacity & (capacity - 1)) == 0):
            raise ValueError("capacity must be a positive power of 2.")
        self._capacity = capacity
        self._value = [neutral_element] * (2 * capacity)
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        if mid + 1 <= start:
            return self._reduce_helper(
                start, end, 2 * node + 1, mid + 1, node_end
            )
        return self._operation(
            self._reduce_helper(start, mid, 2 * node, node_start, mid),
            self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
        )

    def reduce(self, start: int = 0, end: int = None):
        """Apply *operation* over ``arr[start..end]`` (inclusive).

        Parameters
        ----------
        start : int, optional
            Start of the range (default: ``0``).
        end : int or None, optional
            End of the range, inclusive (default: ``capacity - 1``).

        Returns
        -------
        object
            Result of reducing *operation* over the range.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx: int, val) -> None:
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx], self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx: int):
        if not (0 <= idx < self._capacity):
            raise IndexError(f"index {idx} out of range [0, {self._capacity})")
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    """Segment tree with addition as the reduce operation.

    Supports efficient prefix-sum queries and sampling by prefix-sum value.

    Parameters
    ----------
    capacity : int
        Total number of leaf elements (must be a positive power of two).
    """

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity, operation=operator.add, neutral_element=0.0)

    def sum(self, start: int = 0, end: int = None) -> float:
        """Return ``arr[start] + ... + arr[end]``.

        Parameters
        ----------
        start : int, optional
            Start index (default: ``0``).
        end : int or None, optional
            End index, inclusive (default: last element).

        Returns
        -------
        float
        """
        return super().reduce(start, end)

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        """Return the highest index *i* such that ``sum(arr[0..i-1]) <= prefixsum``.

        This is used to sample indices proportionally to their stored values
        (which represent priorities).

        Parameters
        ----------
        prefixsum : float
            Upper bound on the prefix sum. Must satisfy
            ``0 <= prefixsum <= self.sum()``.

        Returns
        -------
        int
            Highest index satisfying the prefix-sum constraint.
        """
        if not (0 <= prefixsum <= self.sum() + 1e-5):
            raise ValueError(
                f"prefixsum {prefixsum} out of range [0, {self.sum()}]"
            )
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    """Segment tree with minimum as the reduce operation.

    Used to efficiently track the minimum priority across all stored
    transitions.

    Parameters
    ----------
    capacity : int
        Total number of leaf elements (must be a positive power of two).
    """

    def __init__(self, capacity: int) -> None:
        super().__init__(
            capacity, operation=min, neutral_element=float("inf")
        )

    def min(self, start: int = 0, end: int = None) -> float:
        """Return ``min(arr[start], ..., arr[end])``.

        Parameters
        ----------
        start : int, optional
            Start index (default: ``0``).
        end : int or None, optional
            End index, inclusive (default: last element).

        Returns
        -------
        float
        """
        return super().reduce(start, end)
