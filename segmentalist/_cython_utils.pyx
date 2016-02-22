#cython: cdivision=True

import random
import numpy as np

cimport cython
cimport numpy as np
from libc.math cimport exp, log


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double logsumexp(double[::1] a):
    """Calculate the logsumexp of the elements of `a`."""
    cdef int j
    cdef double sum_exps = 0.0
    cdef double max_a = a[0]
    cdef int N = a.shape[0]
    
    for j in xrange(1, N):
        if (a[j] > max_a):
            max_a = a[j]
    for j in xrange(N):
        sum_exps += exp(a[j] - max_a)
    return log(sum_exps) + max_a


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sum_doubles(double[::1] y):
    cdef int i
    cdef double sum_y = y[0]
    cdef int N = y.shape[0]
    for i in xrange(1, N):
        sum_y += y[i]
    return sum_y


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.int_t sum_ints(np.int_t[::1] y):
    cdef int i
    cdef np.int_t sum_y = y[0]
    cdef int N = y.shape[0]
    for i in xrange(1, N):
        sum_y += y[i]
    return sum_y


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sum_log(double[::1] y):
    cdef int i
    cdef double sum_log_y = log(y[0])
    cdef int N = y.shape[0]
    for i in xrange(1, N):
        sum_log_y += log(y[i])
    return sum_log_y


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sum_square_a_times_b(double[::1] a, double[::1] b):
    """Every element in `vec` is squared in-place."""
    cdef int i
    cdef int I = a.shape[0]
    cdef double square_sum = 0.0
    for i in xrange(I):
        square_sum += a[i] * a[i] * b[i]
    return square_sum


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int draw(double[::1] p_k):
    """
    Draw from a discrete random variable with mass in vector `p_k`.

    Indices returned are between 0 and len(p_k) - 1.
    """
    cdef int i, J
    cdef double k_uni
    k_uni = random.random()
    J = p_k.shape[0]
    for i in xrange(J):
        k_uni = k_uni - p_k[i]
        if k_uni < 0:
            return i
    return J - 1
