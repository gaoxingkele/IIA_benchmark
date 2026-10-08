"""Skip only zero-contribution None intersections in pinned TAB affiliation.

The reference outer partition constructs G x P entries, most of them None.
Its integral functions give those entries exactly zero contribution (or the
same empty-partition NaN/Inf). Retain every positive intersection, in original
prediction order, and retain all original validation and integral functions.
The inner recall partitions stay original: their singleton-list contract must
not be replaced by an empty sparse list.
"""
from types import FunctionType

from scripts.flow_matching.tab_range_metrics import load_reference


def sparse_outer_partition(events_pred, zones):
    out = []
    cursor = 0
    for left, right in zones:
        while cursor < len(events_pred) and events_pred[cursor][1] <= left:
            cursor += 1
        intersections = []
        index = cursor
        while index < len(events_pred) and events_pred[index][0] < right:
            start, stop = events_pred[index]
            intersection = (max(start, left), min(stop, right))
            if intersection[0] < intersection[1]:
                intersections.append(intersection)
            index += 1
        out.append(intersections)
    return out


def load_sparse_reference(source_root):
    reference = load_reference(source_root)
    original = reference['pr_from_events']
    namespace = dict(original.__globals__)
    namespace['affiliation_partition'] = sparse_outer_partition
    accelerated = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    accelerated.__kwdefaults__ = original.__kwdefaults__
    accelerated.__doc__ = original.__doc__
    return dict(reference, pr_from_events=accelerated)
