# distutils: language = c++
from libcpp.unordered_set cimport unordered_set as cset
import numpy as np
cimport numpy as np
from .tools import is_ndarray, float_type, int_type

ctypedef cset[int] int_set  # unordered_set<int>

cdef extern from "include/randint.h":
    int c_randint_choice(int high, int size, int replace, const float* prob, const int_set* exclusion, int* result)

def pyx_randint_choice(high, size=1, replace=True, p=None, exclusion=None):
    if high <= 1:
        raise ValueError("'high' must be larger than 1.")
    if size <= 0:
        raise ValueError("'size' must be a positive integer.")

    if not isinstance(replace, bool):
        raise TypeError("'replace' must be bool.")
    if p is not None:
        if not is_ndarray(p, float_type):
            p = np.array(p, dtype=float_type)
        if p.ndim != 1:
            raise ValueError("'p' must be a 1-dim array_like")
        if len(p) != high:
            raise ValueError("The length of 'p' must be equal with 'high'.")
    if exclusion is not None and len(exclusion) >= high:
        raise ValueError("The length of 'exclusion' must be smaller than 'high'.")

    len_exclusion = len(exclusion) if exclusion is not None else 0
    if replace is False and (high - len_exclusion <= size):
        raise ValueError("There is not enough integers to be sampled.")

    if isinstance(exclusion, (int, int_type)):
        exclusion = [exclusion]

    cdef int_set* exc_ptr = <int_set*> 0
    cdef int_set _exclusion
    if exclusion is not None:
        _exclusion = exclusion
        exc_ptr = &_exclusion

    cdef float* p_pt = <float*> 0
    if p is not None:
        p_pt = <float *> np.PyArray_DATA(p)  # ndarray

    results = np.zeros(size, dtype=int_type)
    results_pt = <int *>np.PyArray_DATA(results)

    c_randint_choice(high, size, replace, p_pt, exc_ptr, results_pt)

    if len(results) == 1:
        results = results[0]
    return results