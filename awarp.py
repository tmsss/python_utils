import numpy as np
from numba import jit

'''
Python implementation of AWarp algorithm using numba to optimize machine code at runtime.
from https://github.com/mclmza/AWarp
'''

L = 'left'
T = 'top'
D = 'diagonal'
INF = int(1e10)

def rle(series):
    """ 
    Run length encoding for sparse time series to encode zeros as in needed for awarp calculation 
    (https://ieeexplore.ieee.org/document/7837859 | https://github.com/mclmza/AWarp)
    
    args
    ----
    series: sparse times series (e.g. x = [0, 0, 0, 2, 3, 0, 5, 6, 0, 0, 4, 0, 0])

    returns
    ---- 
    array with encoded zeros (e.g. [3 2 3 1 5 6 2 4 2]) """

    # convert to np array
    series = np.array(series)

    # add points to detect inflection on start and end
    series_ = np.concatenate(([1], series, [1]))

    # find zeros and non zeros
    zeros = np.where(series_ == 0)[0]

    if len(zeros) > 0:
        nonzeros = np.where(series_ != 0)[0]

        # detect zero sequencies
        split_zeros = np.where(np.diff(zeros) > 1)[0] + 1

        splitted_zeros = np.split(zeros, split_zeros)

        zero_points = []
        zero_points = np.array(zero_points, dtype=int)
        
        for z in splitted_zeros:
            zero_points = np.append(zero_points, z[-1])

        # detect non-zero sequencies
        nonzero_points = nonzeros[np.where(np.diff(nonzeros) > 1)[0]]

        # concat all splitting points
        split = np.sort(np.concatenate([zero_points, nonzero_points]))

        # avoid splitting on first element
        split = split[split > 0]

        # separate zero sequencies from non-zero sequencies
        splitted_series = np.split(series, split)

        # initialize empty array
        rle = []
        rle = np.array(rle, dtype=int)

        # encode zeros
        for s in splitted_series:
            # if it is a zero sequence enconde the lenght of the sequence
            if np.sum(s) == 0:
                rle = np.append(rle, [len(s)], axis=0)
            else:
                rle = np.concatenate([rle, s])

        # remove zeros in the end
        rle = rle[rle > 0]

        return rle
    else:
        return series


@jit(nopython=True)
def ub_costs(a, b, case):
    if a > 0 and b > 0:
        return (a - b) ** 2
    elif b < 0 < a:
        if case == L:
            return a ** 2
        else:
            return -b * a ** 2
    elif b > 0 > a:
        if case == T:
            return b ** 2
        else:
            return -a * (b ** 2)
    else:
        return 0


@jit(nopython=True)
def ub_costs_constrained(a, b, mode, w, gap):
    if a > 0 and b > 0 and gap <= w:
        return (a - b) ** 2
    elif a < 0 and b < 0:
        return 0
    else:
        if mode == D:
            if b < 0 < a:
                return -b * (a**2)
            elif a < 0 < b:
                return -a * (b**2)
            else:
                return int(INF)
        elif mode == L:
            if b < 0 < a and gap <= w:
                return -b * (a**2)
            elif a < 0 < b:
                return b ** 2
            else:
                return int(INF)
        elif mode == T:
            if b < 0 < a:
                return a**2
            elif a < 0 < b and gap <= w:
                return -a * (b**2)
            else:
                return int(INF)


@jit(nopython=True)
def compute_awarp(d, x, y):
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            if i > 0 and j > 0:
                a_d = d[i, j] + ub_costs(x[i], y[j], 'diagonal')
            else:
                a_d = d[i, j] + (x[i] - y[j]) ** 2
            a_l = d[i+1, j] + ub_costs(x[i], y[j], 'top')
            a_t = d[i, j+1] + ub_costs(x[i], y[j], 'left')
            d[i+1, j+1] = min(a_d, a_t, a_l)

@jit(nopython=True)
def compute_awarp_constrained(d, x, y, w, t_x, t_y):
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            gap = np.absolute(t_x[i] - t_y[j])
            if gap > w and ((j > 0 and t_y[j-1] - t_x[i] > w) or (i > 0 and t_x[i-1] - t_y[j] > w)):
                d[i+1, j+1] = int(INF)
            else:
                if i > 0 and j > 0:
                    a_d = d[i, j] + ub_costs_constrained(x[i], y[j], D, w, gap)
                else:
                    a_d = d[i, j] + (x[i] - y[j]) ** 2
                a_l = d[i+1, j] + ub_costs_constrained(x[i], y[j], L, w, gap)
                a_t = d[i, j+1] + ub_costs_constrained(x[i], y[j], T, w, gap)
                d[i+1, j+1] = min(a_d, a_t, a_l)


def awarp(x, y, w=0):

    # run length enconde series
    x = rle(x)
    y = rle(y)

    d = np.zeros((x.shape[0] + 1, y.shape[0] + 1)).astype(int)
    d[:, 0] = int(INF)
    d[0, :] = int(INF)
    d[0, 0] = 0

    if w > 0:
        t_x = np.zeros(x.shape[0] + 1).astype(int)
        t_y = np.zeros(y.shape[0] + 1).astype(int)

        iit = 0
        for i in range(x.shape[0]):
            if x[i] > 0:
                iit += 1
            else:
                iit += abs(x[i])
            t_x[i] = iit
        t_x[-1] = iit + 1

        iit = 0
        for i in range(y.shape[0]):
            if y[i] > 0:
                iit += 1
            else:
                iit += abs(y[i])
            t_y[i] = iit
        t_y[-1] = iit + 1
        compute_awarp_constrained(d, x, y, w, t_x, t_y)
    else:
        compute_awarp(d, x, y)

    return np.sqrt(d[-1, -1])
