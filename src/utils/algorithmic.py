from bisect import bisect_left

from typing import Protocol, Sequence, TypeVar


class Comparable(Protocol):
    def __lt__(self, other) -> bool: ...
    def __eq__(self, other) -> bool: ...


T = TypeVar('T', bound=Comparable)


def get_closest_value_in_sorted_sequence(value: T, seq: Sequence[T], error_on_out_of_bounds: bool = False) -> T:
    """
    Assumes the list is sorted, and `value` is inside the list's range. Then, returns
    the closest value to `value` inside the list using binary search.
    """
    pos = bisect_left(seq, value)
    if pos == 0:
        if error_on_out_of_bounds and value < seq[0]:
            raise ValueError(f"Value {value} must be inside the list's range [{seq[0]}, {seq[-1]}]")
        return seq[0]
    if pos == len(seq):
        if error_on_out_of_bounds and value > seq[-1]:
            raise ValueError(f"Value {value} must be inside the list's range [{seq[0]}, {seq[-1]}]")
        return seq[-1]
    # Pick the closest value out of the two closest:
    before = seq[pos - 1]
    after = seq[pos]
    if after - value < value - before:
        return after
    else:
        return before
