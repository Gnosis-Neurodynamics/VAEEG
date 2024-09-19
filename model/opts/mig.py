import math
from collections import Counter

import numpy as np
from ..opts import type_assert


@type_assert(seq1=np.ndarray, seq2=np.ndarray, eps=float)
def discrete_mutual_info(seq1, seq2, eps=1.0e-8):
    """
    :param seq1: 1d np.ndarray, an array of ints, shape (batch,)
    :param seq2: 1d np.ndarray, an array of ints, shape (batch,)
    :param eps:
    :return:
    """
    if seq1.ndim != 1 or "int" not in seq1.dtype.name:
        raise ValueError("seq1 must be a 1D Numpy array of integers!")

    if seq2.ndim != 1 or "int" not in seq2.dtype.name:
        raise ValueError("seq2 must be a 1D Numpy array of integers!")

    if seq1.shape != seq2.shape:
        raise ValueError("seq1 and seq2 must have the same shape")

    n = seq1.shape[0]

    if n == 0:
        raise ValueError("seq1 and seq2 can't be empty.")

    n *= 1.0

    a_counter = Counter(seq1.tolist())
    b_counter = Counter(seq2.tolist())
    ab_counter = Counter(list(zip(seq1, seq2)))

    mi = 0.0
    for (ai, bi), num in ab_counter.items():
        num_ai = a_counter[ai] * 1.0
        num_bi = b_counter[bi] * 1.0
        num *= 1.0
        mi += num / n * (math.log(max(n * num, eps)) - math.log(max(num_ai * num_bi, eps)))

    return mi


@type_assert(seq=np.ndarray)
def discrete_entropy(seq):
    """

    :param seq: 1d np.ndarray, an array of ints, shape (batch,)
    :return:
    """

    if seq.ndim != 1 or "int" not in seq.dtype.name:
        raise ValueError("seq must be a 1D Numpy array of integers!")

    n = seq.shape[0]

    if n == 0:
        raise ValueError("seq can't be empty.")

    n *= 1.0

    a_counter = Counter(seq.tolist())

    ent = 0
    for ai, num_ai in a_counter.items():
        ai_prob = num_ai / n
        ent -= ai_prob * math.log(ai_prob)

    return ent


@type_assert(x=np.ndarray, num_bins=int)
def histogram_discretize(x, num_bins):
    """

    :param x: numpy.ndarray of floats
    :param num_bins: int, num_bins + 2 in the result
    :return:
    """

    if x.ndim != 1 or "float" not in x.dtype.name:
        raise ValueError("x must be a 1D Numpy array of float!")

    if x.shape[0] == 0:
        raise ValueError("x can't be empty.")

    # robust for outliers, using +- 3 sigma for bins
    q = np.quantile(x, [0.005, 0.995])
    flag = np.logical_and(x >= q[0], x <= q[1])

    sx = x[flag]
    u = np.mean(sx)
    std = np.std(sx)

    hist, bin_edges = np.histogram(x, bins=num_bins, range=(u - 3 * std, u + 3 * std))
    x_disc = np.digitize(x, bin_edges)
    return x_disc


@type_assert(z=np.ndarray, y=np.ndarray, num_bins=int, eps=float)
def compute_mig(z, y, num_bins=40, eps=1.0e-8):
    """
    Compute MIG
    Sepliarskaia, A., & Kiseleva, J. (n.d.). Evaluating Disentangled Representations. 1–16.

    :param z: numpy.ndarray, 2d, (z_dim, n_samples)
    :param y: numpy.ndarray, 2d, (z_y, n_samples)
    :param num_bins: int, default 40. num_bins + 2 will be used at last.
    :param eps: float,
    :return:
    """
    if z.ndim != 2 or "float" not in z.dtype.name:
        raise ValueError("z must be a 2D Numpy array of float!")

    if y.ndim != 2 or "float" not in y.dtype.name:
        raise ValueError("y must be a 2D Numpy array of float!")

    if z.shape[1] != y.shape[1]:
        raise ValueError("The shape of z and y on axis 1 must be the same.")

    n_latent = z.shape[0]

    dz = np.apply_along_axis(lambda iv: histogram_discretize(iv, num_bins), 1, z)
    dy = np.apply_along_axis(lambda iv: histogram_discretize(iv, num_bins), 1, y)

    mi_zy = []

    for i in range(n_latent):
        zi = dz[i]
        mi_i = np.apply_along_axis(lambda yj: discrete_mutual_info(zi, yj, eps), 1, dy)
        mi_zy.append(mi_i)

    mi_zy = np.stack(mi_zy, axis=0)
    hy = np.apply_along_axis(discrete_entropy, 1, dy)

    mi_sorted = np.sort(mi_zy, axis=0)[::-1]
    mi_gap_y = np.divide(mi_sorted[0, :] - mi_sorted[1, :], hy)
    mig = np.mean(mi_gap_y)
    return mig


@type_assert(z=np.ndarray, y=np.ndarray, num_bins=int, eps=float)
def compute_dcimig(z, y, num_bins=40, eps=1.0e-8):
    """
    Compute DCIMIG
    Sepliarskaia, A., & Kiseleva, J. (n.d.). Evaluating Disentangled Representations. 1–16.

    :param z: numpy.ndarray, 2d, (z_dim, n_samples)
    :param y: numpy.ndarray, 2d, (z_y, n_samples)
    :param num_bins: int, default 40. num_bins + 2 will be used at last.
    :param eps: float,
    :return:
    """
    if z.ndim != 2 or "float" not in z.dtype.name:
        raise ValueError("z must be a 2D Numpy array of float!")

    if y.ndim != 2 or "float" not in y.dtype.name:
        raise ValueError("y must be a 2D Numpy array of float!")

    if z.shape[1] != y.shape[1]:
        raise ValueError("The shape of z and y on axis 1 must be the same.")

    n_latent = z.shape[0]

    dz = np.apply_along_axis(lambda iv: histogram_discretize(iv, num_bins), 1, z)
    dy = np.apply_along_axis(lambda iv: histogram_discretize(iv, num_bins), 1, y)

    mi_zy = []

    for i in range(n_latent):
        zi = dz[i]
        mi_i = np.apply_along_axis(lambda yj: discrete_mutual_info(zi, yj, eps), 1, dy)
        mi_zy.append(mi_i)

    mi_zy = np.stack(mi_zy, axis=0)
    hy = np.apply_along_axis(discrete_entropy, 1, dy)
    hy_sum = np.sum(hy)

    mi_sorted = np.sort(mi_zy, axis=1)[:, ::-1]
    di = mi_sorted[:, 0] - mi_sorted[:, 1]

    kj = np.zeros_like(mi_zy, dtype=np.bool)

    ji = np.argmax(mi_zy, axis=1)
    for r, c in enumerate(ji):
        kj[r, c] = True

    dyj = np.apply_along_axis(lambda t: max(di[t]) if len(di[t]) > 0 else 0.0, 0, kj)

    dci_mig = np.sum(dyj) / hy_sum
    return dci_mig
