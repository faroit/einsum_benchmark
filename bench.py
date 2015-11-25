import numpy as np

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


A = nnrandn((40, 40, 100, 10))
X = nnrandn((40, 10))
Z = nnrandn((40, 10))

# V(a,b,f,t,c) = A(a,b,f,j)X(t,j)Z(c,j)
V = eps+np.einsum('abfj,tj,cj->abftc', A, X, Z)
