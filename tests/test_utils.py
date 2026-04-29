"""Tests for ddpg_per.utils (LinearSchedule and SegmentTree classes)."""

import pytest
from ddpg_per.utils import LinearSchedule, SumSegmentTree, MinSegmentTree


class TestLinearSchedule:
    def test_initial_value(self):
        sched = LinearSchedule(100, final_p=1.0, initial_p=0.4)
        assert sched.value(0) == pytest.approx(0.4)

    def test_final_value(self):
        sched = LinearSchedule(100, final_p=1.0, initial_p=0.4)
        assert sched.value(100) == pytest.approx(1.0)

    def test_midpoint(self):
        sched = LinearSchedule(100, final_p=1.0, initial_p=0.4)
        assert sched.value(50) == pytest.approx(0.7)

    def test_clamped_after_end(self):
        sched = LinearSchedule(100, final_p=1.0, initial_p=0.4)
        assert sched.value(200) == pytest.approx(1.0)

    def test_default_initial_p(self):
        sched = LinearSchedule(10, final_p=0.0)
        assert sched.value(0) == pytest.approx(1.0)


class TestSumSegmentTree:
    def test_basic_sum(self):
        tree = SumSegmentTree(4)
        tree[0] = 1.0
        tree[1] = 2.0
        tree[2] = 3.0
        tree[3] = 4.0
        assert tree.sum() == pytest.approx(10.0)

    def test_partial_sum(self):
        tree = SumSegmentTree(4)
        tree[0] = 1.0
        tree[1] = 2.0
        tree[2] = 3.0
        tree[3] = 4.0
        # end is exclusive: sum(0, 3) covers indices 0, 1, 2 → 1+2+3=6
        assert tree.sum(0, 3) == pytest.approx(6.0)

    def test_single_element(self):
        tree = SumSegmentTree(4)
        tree[2] = 5.0
        assert tree.sum(2, 3) == pytest.approx(5.0)

    def test_find_prefixsum_idx_first(self):
        tree = SumSegmentTree(4)
        tree[0] = 1.0
        tree[1] = 2.0
        tree[2] = 3.0
        tree[3] = 4.0
        # prefixsum=0 should always return index 0
        assert tree.find_prefixsum_idx(0.5) == 0

    def test_find_prefixsum_idx_last(self):
        tree = SumSegmentTree(4)
        tree[0] = 1.0
        tree[1] = 2.0
        tree[2] = 3.0
        tree[3] = 4.0
        assert tree.find_prefixsum_idx(9.5) == 3

    def test_invalid_capacity(self):
        with pytest.raises(ValueError):
            SumSegmentTree(3)  # not a power of two

    def test_out_of_range_getitem(self):
        tree = SumSegmentTree(4)
        with pytest.raises(IndexError):
            _ = tree[4]

    def test_invalid_prefixsum(self):
        tree = SumSegmentTree(4)
        tree[0] = 1.0
        with pytest.raises(ValueError):
            tree.find_prefixsum_idx(-0.1)


class TestMinSegmentTree:
    def test_basic_min(self):
        tree = MinSegmentTree(4)
        tree[0] = 3.0
        tree[1] = 1.0
        tree[2] = 4.0
        tree[3] = 2.0
        assert tree.min() == pytest.approx(1.0)

    def test_partial_min(self):
        tree = MinSegmentTree(4)
        tree[0] = 3.0
        tree[1] = 1.0
        tree[2] = 4.0
        tree[3] = 2.0
        assert tree.min(2, 4) == pytest.approx(2.0)

    def test_default_neutral_element(self):
        tree = MinSegmentTree(4)
        assert tree.min() == float("inf")
