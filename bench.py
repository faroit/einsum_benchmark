import numpy as np
import timeit

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


A = nnrandn((2000, 100))
B = nnrandn((2000, 100))


def run():
    V = np.einsum('ak,bk->ab', A, B)
    return V

times = timeit.Timer(run).timeit(number=100)

print times
