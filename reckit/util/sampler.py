from reckit.cython.cpr import CyCPRSampler
from math import ceil
import numpy as np

def invert_dict(d, sort=False):
    inverse = {}
    for key in d:
        for value in d[key]:
            if value not in inverse:
                inverse[value] = [key]
            else:
                inverse[value].append(key)
    return inverse

def batch_iterator(data, batch_size, drop_last=False):
    """Generate batches.
    Args:
        data (list or numpy.ndarray): Input data.
        batch_size (int): Size of each batch except for the last one.
    """
    length = len(data)
    if drop_last:
        n_batch = length // batch_size
    else:
        n_batch = ceil(length / batch_size)
    for i in range(n_batch):
        yield data[i * batch_size : (i + 1) * batch_size]

class CPRSampler(object):
    def __init__(self, dataset, sample_ratio, sample_rate, batch_size=1024, n_thread=4, k_interact=None, max_k_interact=3, drop_last=False):
        """
        CPR采样
        :param dataset: Interaction
        :param batch_size:
        :param n_thread:
        """
        self.batch_size = batch_size
        self.n_thread = n_thread
        self.drop_last = drop_last

        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.num_ratings = dataset.num_ratings

        self.user_pos_dict = dataset.train_data.to_user_dict()  # OrderedDict(userID: DataFrame[items])
        self.train = [self.user_pos_dict[u] if u in self.user_pos_dict.keys() else [] for u in range(self.num_users)]
        self.train_inverse = invert_dict(self.user_pos_dict)
        self.train_inverse = [self.train_inverse[i] if i in self.train_inverse else [] for i in range(self.num_items)]

        self.u_interacts, self.i_interacts = [], []
        for u, items in enumerate(self.train):
            for i in items:
                self.u_interacts.append(u)
                self.i_interacts.append(i)
        self.u_interacts = np.array(self.u_interacts, dtype=np.int32)
        self.i_interacts = np.array(self.i_interacts, dtype=np.int32)

        self.n_step = self.num_ratings // self.batch_size
        if k_interact is None:
            ratios = np.power(sample_ratio, np.arange(max_k_interact - 2, -1, -1))
        else:
            ratios = np.array([0]*(k_interact-2) + [1])
        batch_sizes = np.round(batch_size / np.sum(ratios) * ratios).astype(np.int32)
        batch_sizes[-1] = batch_size - np.sum(batch_sizes[:-1])

        self.batch_sample_sizes = np.ceil(np.array(batch_sizes)*sample_rate).astype(np.int32)
        self.batch_total_sample_sizes = self.batch_sample_sizes * np.arange(2, len(self.batch_sample_sizes)+2)
        self.batch_sample_size = np.sum(self.batch_sample_sizes)
        self.sample_size = self.n_step * self.batch_sample_size

        self.batch_choice_sizes = 2 * self.batch_sample_sizes  # gamma默认为2
        self.choice_size = 2 * self.sample_size

        self.users = np.empty(self.sample_size, dtype=np.int32)
        self.items = np.empty(self.sample_size, dtype=np.int32)

        self.cpr_sampler = CyCPRSampler(
            self.train,
            self.u_interacts,
            self.i_interacts,
            self.users,  # 传出参数
            self.items,  # 传出参数
            self.n_step,
            self.batch_sample_sizes,
            self.n_thread
        )

    def sample(self):
        interact_idx = np.random.choice(self.num_ratings, size=self.choice_size).astype(np.int32)
        if self.cpr_sampler.sample(interact_idx, self.batch_choice_sizes) == -1:
            raise RuntimeError("choice_size is not large enough")
        
        return zip(
            batch_iterator(self.users, self.batch_sample_size),
            batch_iterator(self.items, self.batch_sample_size),
        )