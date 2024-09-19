import itertools
from copy import deepcopy
from functools import partial

import numpy as np
from interval3 import Interval

from .check import check_type

__all__ = ["Interval", "IntervalSet", "find_continuous_area_1d", "find_continuous_area_2d",
           "merge_continuous_area", "merge_continuous_area_multi"]


class BaseIntervalSet(object):
    def __init__(self, items=None):
        check_type("items", items, [type(None), list])
        if items is None:
            self.intervals = []
        else:
            self.intervals = _union(deepcopy(items))

        self.intervals.sort()

    def __len__(self):
        return len(self.intervals)

    def __repr__(self):
        return "IntervalSet([%s])" % (
            ", ".join(repr(i) for i in self.intervals),)

    def __str__(self):
        if len(self.intervals) == 0:
            range_str = "<Empty>"
        else:
            range_str = ",".join([str(r) for r in self.intervals])
        return range_str

    def __getitem__(self, index):
        try:
            return self.intervals[index]
        except IndexError:
            raise IndexError("Index is out of range")

    def __iter__(self):
        return self.intervals.__iter__()

    def __contains__(self, item):
        check_type("item", item, [Interval])

        result = False
        for r in self.intervals:
            if item in r:
                result = True
                break
        return result

    def lower_bound(self):
        if len(self.intervals) > 0:
            return self.intervals[0].lower_bound
        else:
            raise IndexError("The BaseIntervalSet is empty")

    def upper_bound(self):
        if len(self.intervals) > 0:
            return self.intervals[-1].upper_bound
        else:
            raise IndexError("The BaseIntervalSet is empty")

    def lower_closed(self):
        if len(self.intervals) > 0:
            return self.intervals[0].lower_closed
        else:
            raise IndexError("The BaseIntervalSet is empty")

    def upper_closed(self):
        if len(self.intervals) > 0:
            return self.intervals[0].upper_closed
        else:
            raise IndexError("The BaseIntervalSet is empty")

    def bounds(self):
        if len(self.intervals) == 0:
            result = Interval.none()
        else:
            result = Interval(
                self.lower_bound(), self.upper_bound(),
                lower_closed=self.lower_closed(),
                upper_closed=self.upper_closed())
        return result

    def copy(self):
        return deepcopy(self)


def _issubset(a, b):
    check_type("a", a, [BaseIntervalSet])
    check_type("b", b, [BaseIntervalSet])

    result = True
    for i in a:
        if i not in b:
            result = False
            break
    return result


class IntervalSet(BaseIntervalSet):
    def __init__(self, items=None):
        BaseIntervalSet.__init__(self, items)

    def issubset(self, other):
        return _issubset(self, other)

    def __eq__(self, other):
        check_type("other", other, [BaseIntervalSet])
        result = _issubset(self, other) and _issubset(other, self)
        return result

    def __ne__(self, other):
        check_type("other", other, [BaseIntervalSet])
        return not (self == other)

    def difference(self, other):
        check_type("other", other, [BaseIntervalSet])
        items = []
        items.extend([(i, 0) for i in deepcopy(self.intervals)])
        items.extend([(i, 1) for i in deepcopy(other.intervals)])
        items.sort()

        res = []
        pre = None
        for cur in items:
            if pre is None:
                pre = cur
            else:
                if pre[1] == 0 and cur[1] == 0:
                    res.append(pre[0])
                    pre = cur
                elif pre[1] == 0 and cur[1] == 1:
                    if cur[0].lower_bound > pre[0].upper_bound:
                        res.append(pre[0])
                        pre = cur
                    elif cur[0].lower_bound == pre[0].upper_bound:
                        if cur[0].lower_closed and pre[0].upper_closed:
                            pre[0].upper_closed = False
                        res.append(pre[0])
                        pre = cur
                    else:
                        # update pre on lower end
                        if cur[0].lower_bound == pre[0].lower_bound:
                            if pre[0].lower_closed and (not cur[0].lower_closed):
                                # pre lower point
                                res.append(Interval(lower_bound=pre[0].lower_bound,
                                                    upper_bound=pre[0].lower_bound,
                                                    lower_closed=True,
                                                    upper_closed=True))
                                pre[0].lower_closed = False
                        else:
                            res.append(Interval(lower_bound=pre[0].lower_bound,
                                                upper_bound=cur[0].lower_bound,
                                                lower_closed=pre[0].lower_closed,
                                                upper_closed=not cur[0].lower_closed))
                            pre[0].lower_bound = cur[0].lower_bound
                            pre[0].lower_closed = cur[0].lower_closed

                        # update pre on upper end
                        if cur[0].upper_bound == pre[0].upper_bound:
                            if pre[0].upper_closed and (not cur[0].upper_closed):
                                # pre upper point
                                res.append(Interval(lower_bound=pre[0].upper_bound,
                                                    upper_bound=pre[0].upper_bound,
                                                    lower_closed=True,
                                                    upper_closed=True))
                            pre = None
                        elif cur[0].upper_bound < pre[0].upper_bound:
                            pre[0].lower_bound = cur[0].upper_bound
                            pre[0].lower_closed = not cur[0].upper_closed
                        else:
                            cur[0].lower_bound = pre[0].upper_bound
                            cur[0].lower_closed = not pre[0].lower_closed
                            pre = cur
                elif pre[1] == 1 and cur[1] == 0:
                    if cur[0].lower_bound > pre[0].upper_bound:
                        pre = cur
                    elif cur[0].lower_bound == pre[0].upper_bound:
                        if cur[0].lower_closed and pre[0].upper_closed:
                            cur[0].lower_closed = False
                        pre = cur
                    else:
                        if cur[0].upper_bound > pre[0].upper_bound:
                            cur[0].lower_bound = pre[0].upper_bound
                            cur[0].lower_closed = not pre[0].upper_closed
                            pre = cur
                        elif cur[0].upper_bound == pre[0].upper_bound:
                            if cur[0].upper_closed and (not pre[0].upper_closed):
                                # cur a point left
                                pre = (Interval(lower_bound=cur[0].upper_bound,
                                                upper_bound=cur[0].upper_bound,
                                                lower_closed=True,
                                                upper_closed=True), 0)
                            elif (not cur[0].upper_closed) and pre[0].upper_closed:
                                # pre a point left
                                pre = (Interval(lower_bound=cur[0].upper_bound,
                                                upper_bound=cur[0].upper_bound,
                                                lower_closed=True,
                                                upper_closed=True), 1)
                            else:
                                pre = None
                        else:
                            pre[0].lower_bound = cur[0].upper_bound
                            pre[0].lower_closed = not cur[0].upper_closed
                else:
                    pre = cur

        if pre is not None and pre[1] == 0:
            res.append(pre[0])

        return self.__class__(res)

    def union(self, other):
        check_type("other", other, [BaseIntervalSet])
        items = deepcopy(self.intervals + other.intervals)
        res = _union(items)
        return self.__class__(res)

    def intersection(self, other):
        check_type("other", other, [BaseIntervalSet])
        return self.difference(self.difference(other))

    def scale(self, scale_value):
        check_type("scale_value", scale_value, [float])
        if scale_value <= 0.0:
            raise ValueError("`scale_value` must be greater than 0, but got %s instead." % scale_value)

        res = []
        for item in self.intervals:
            res.append(Interval(lower_bound=item.lower_bound / scale_value, lower_closed=item.lower_closed,
                                upper_bound=item.upper_bound / scale_value, upper_closed=item.upper_closed))
        return self.__class__(res)


def _union(items):
    check_type("items", items, [list])
    res = []
    items.sort()

    msg = "Element of `items` must be {support_type}, but got {cur_type} instead: {value}"

    for cur in items:
        if not isinstance(cur, Interval):
            raise TypeError(msg.format(
                support_type=repr(Interval),
                cur_type=repr(type(cur)),
                value=repr(cur)))

        if cur.lower_bound != cur.upper_bound or (cur.lower_closed and cur.upper_closed):
            if not res:
                res.append(cur)
            else:
                # not a copy, just an alias
                pre = res[-1]

                if cur.lower_bound > pre.upper_bound:
                    res.append(cur)
                elif cur.lower_bound == pre.upper_bound:
                    if cur.lower_closed or pre.upper_closed:
                        pre.upper_bound = cur.upper_bound
                        pre.upper_closed = cur.upper_closed
                    else:
                        res.append(cur)
                else:
                    if cur.upper_bound > pre.upper_bound:
                        pre.upper_bound = cur.upper_bound
                        pre.upper_closed = cur.upper_closed
                    elif cur.upper_bound == pre.upper_bound:
                        pre.upper_closed = cur.upper_closed or pre.upper_closed
    return res


def find_continuous_area_1d(mask, area_threshold=0):
    """
    Find continuous areas where mask is True.

    Args:
        mask: numpy.ndarray, bool, 1d (d1, )
        area_threshold: int, optional, default 0
            Areas whose range are no less than area_threshold are kept.

    Returns:
        IntervalSet

    """
    check_type("mask", mask, [np.ndarray])
    check_type("area_threshold", area_threshold, [int])

    if mask.dtype != bool:
        raise TypeError("The data type of `mask` must be numpy.bool, but got %s instead." % mask.dtype)

    if mask.ndim != 1:
        raise ValueError("`mask` must be 1d-numpy.ndarray, "
                         "but got {dimension}d instead.".format(dimension=mask.ndim))

    if area_threshold < 0:
        raise ValueError("area_threshold must be integer no less than 0, but got %s" % area_threshold)

    res = []

    for k, g in itertools.groupby(enumerate(mask), key=lambda x: x[1]):
        if k:
            index, _ = list(zip(*g))
            if index[-1] - index[0] >= area_threshold:
                res.append(Interval(lower_bound=index[0], upper_bound=index[-1],
                                    lower_closed=True, upper_closed=True))
    return IntervalSet(res)


def find_continuous_area_2d(mask, area_threshold=0):
    """
    Find continuous areas where mask is True on the last axis.

    Args:
        mask: numpy.ndarray, bool, 2d (d1,d2)
        area_threshold: int, optional, default 0
            Areas whose range are no less than area_threshold are kept.

    Returns:
        a list of IntervalSet, its length is equal to d1

    """
    check_type("mask", mask, [np.ndarray])
    check_type("area_threshold", area_threshold, [int])

    if mask.dtype != bool:
        raise TypeError("The data type of `mask` must be numpy.bool, but got %s instead." % mask.dtype)

    if mask.ndim != 2:
        raise ValueError("`mask` must be 2d-numpy.ndarray, "
                         "but got {dimension}d instead.".format(dimension=mask.ndim))

    if area_threshold < 0:
        raise ValueError("area_threshold must be integer no less than 0, but got %s" % area_threshold)

    func = partial(find_continuous_area_1d, area_threshold=area_threshold)

    res = list(map(func, mask))
    return res


def merge_continuous_area(x, threshold):
    """

    Args:
        x: IntervalSet,
            input IntervalSet
        threshold: float or int, > 0.0
            If two neighbor intervals' distance <= threshold, merge them with the lower bound
            of the former interval and the upper bound of the latter interval.

    Returns:
        IntervalSet
    """
    check_type("x", x, [IntervalSet])
    check_type("threshold", threshold, [float, int])

    if threshold <= 0.0:
        raise ValueError("threshold must be float or int > 0.0, but got %s instead" % threshold)

    n = len(x)
    if n <= 1:
        return x

    res = []

    pre = x[0]
    for i in range(1, n):
        cur = x[i]
        if cur.lower_bound - pre.upper_bound <= threshold:
            # merge
            pre = Interval(lower_bound=pre.lower_bound, lower_closed=pre.lower_closed,
                           upper_bound=cur.upper_bound, upper_closed=cur.upper_closed)
        else:
            res.append(pre)
            pre = cur

    res.append(pre)
    return IntervalSet(res)


def merge_continuous_area_multi(xs, threshold):
    """
    Union the IntervalSet items in `xs` first, then apply `merge_continuous_area`

    Args:
        xs: a list of IntervalSet,
            input IntervalSet
        threshold: float or int, > 0.0
            If two neighbor intervals' distance <= threshold, merge them with the lower bound
            of the former interval and the upper bound of the latter interval.

    Returns:
        IntervalSet
    """
    check_type("xs", xs, [list])

    x = []
    for item in xs:
        if not isinstance(item, IntervalSet):
            raise TypeError("Element of `item` must be InterSet data type,"
                            " but got {v_type}: {value} ".format(v_type=repr(type(item)),
                                                                 value=item))
        else:
            x.extend(item.intervals)

    return merge_continuous_area(IntervalSet(x), threshold)



def stride_data(x, n_per_seg, n_overlap):
    """clips data to interest segment sequences
    """
    if n_per_seg == 1 and n_overlap == 0:
        result = x[..., np.newaxis]
    else:
        step = n_per_seg - n_overlap
        shape = x.shape[:-1]+((x.shape[-1]-n_overlap)//step, n_per_seg)
        strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return result
