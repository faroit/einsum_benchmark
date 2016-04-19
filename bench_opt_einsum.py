from __future__ import print_function
import numpy as np
import timeit
import functools
import csv


def parafac2(n, r=10000):
    A = np.random.randn(n, r)
    B = np.random.randn(n, r)

    return opt_einsum.contract('aj,bj->ab', A, B)


def parafac5(n, r=10):
    A = np.random.randn(n, r)
    B = np.random.randn(n, r)
    C = np.random.randn(n, r)
    D = np.random.randn(n, r)
    E = np.random.randn(n, r)

    return opt_einsum.contract('aj,bj,cj,dj,ej->abcde', A, B, C, D, E)


def commonfate(n, r=10):
    A = np.random.random((n, n, n, r))
    H = np.random.random((n, r))
    C = np.random.random((n, r))

    return opt_einsum.contract('abfj,tj,cj->abftc', A, H, C)


def bench(n, reps, func):
    return timeit.Timer(
        functools.partial(func, n)
    ).timeit(number=reps) / reps


with open('py_opt.csv', 'wb') as csvfile:
    benchwriter = csv.writer(
        csvfile, delimiter=' ',
        quotechar='|', quoting=csv.QUOTE_MINIMAL
    )

    for i in range(10, 80, 10):
        benchwriter.writerow(['parafac2', i, bench(i, 3, parafac2)])
        benchwriter.writerow(['parafac5', i, bench(i, 3, parafac5)])
        benchwriter.writerow(['commonfate', i, bench(i, 3, commonfate)])
