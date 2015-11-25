import numpy as np
import timeit
from parafac_fast import parafac

eps = 1e-10


def nnrandn(shape):
    """generates randomly a nonnegative ndarray of given shape
    Parameters
    ----------
    shape : tuple
        The shape
    Returns
    -------
    out : array of given shape
        The non-negative random numbers
    """
    return np.abs(np.random.randn(*shape))


A = nnrandn((1000, 50))
B = nnrandn((1000, 50))

factors = [A, B]


def run():
    V = parafac(factors)
    return V

times = timeit.Timer(run).timeit(number=100)

print times
