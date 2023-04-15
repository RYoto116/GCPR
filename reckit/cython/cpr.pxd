from libcpp.vector cimport vector
from libcpp.unordered_set cimport unordered_set

cdef extern from "include/cpr.h":
    cdef cppclass CppCPRSampler:
        CppCPRSampler() except +
        CppCPRSampler(vector[unordered_set[int]] train,
                    int *u_interacts,
                    int *i_interacts,
                    int *users,
                    int *items,
                    int n_step,
                    int *batch_sample_sizes,
                    int sizes_len,
                    int n_thread) except +

        int Sample(int *interact_idx, int interact_idx_len, int *batch_choice_sizes)
