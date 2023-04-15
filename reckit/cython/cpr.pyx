# distutils: language = c++
import numpy as np
cimport numpy as np
from .cpr cimport CppCPRSampler

cdef class CyCPRSampler:
    cdef CppCPRSampler c_cross_sampler
    
    def __init__(self, 
                    vector[unordered_set[int]] train,
                    np.ndarray[int, ndim=1] u_interacts,
                    np.ndarray[int, ndim=1] i_interacts,
                    np.ndarray[int, ndim=1] users,
                    np.ndarray[int, ndim=1] items,
                    int n_step,
                    np.ndarray[int, ndim=1] batch_sample_sizes,
                    int n_thread):
        u_interacts = np.ascontiguousarray(u_interacts)
        i_interacts = np.ascontiguousarray(i_interacts)
        batch_sample_sizes = np.ascontiguousarray(batch_sample_sizes)
        self.c_cross_sampler = CppCPRSampler(train, 
                                            &u_interacts[0], 
                                            &i_interacts[0], 
                                            &users[0], 
                                            &items[0],
                                            n_step, 
                                            &batch_sample_sizes[0], 
                                            batch_sample_sizes.shape[0], 
                                            n_thread)
    
    def sample(self, np.ndarray[int, ndim=1] interact_idx, np.ndarray[int, ndim=1] batch_choice_sizes):
        interact_idx = np.ascontiguousarray(interact_idx)
        return self.c_cross_sampler.Sample(&interact_idx[0], interact_idx.shape[0], &batch_choice_sizes[0])
