# distutils: language = c++
from libcpp.unordered_set cimport unordered_set as cset
from libcpp.vector cimport vector as cvector
import numpy as np
cimport numpy as np
from .tools import float_type

ctypedef cset[int] int_set

cdef extern from "include/evaluate.h":
    void cpp_evaluate_matrix(float *rating_matrix, int rating_len,
                             cvector[int_set] &test_items,
                             cvector[int] metric,
                             cvector[int] &item_degrees, 
                             int topk,
                             int thread_num, float *results_pt)

def eval_score_matrix(score_matrix, test_items, metric, item_degrees, topk, thread_num):
    rating_len = np.shape(score_matrix)[-1]
    user_num = len(test_items)
    cdef float *scores_pt = <float *>np.PyArray_DATA(score_matrix)
    cdef cvector[int_set] test_items_vec = test_items
    cdef cvector[int] metric_vec = metric
    cdef cvector[int] degree_vec = item_degrees

    metric_nums = len(metric)
    results = np.zeros([user_num, metric_nums * topk], dtype=float_type)
    results_pt = <float *>np.PyArray_DATA(results)
    cpp_evaluate_matrix(scores_pt, rating_len, test_items_vec, metric_vec, degree_vec, topk, thread_num, results_pt)

    return results