# 采样负样本
import numpy as np
from math import ceil
from collections import defaultdict, Iterable
from reckit.random import randint_choice
from reckit.dataiterator import DataIterator
from reckit.cython.sampler import CyCPRSampler

class Sampler(object):
    """所有负样本采样器的基类"""
    """初始化采样器"""
    def __init__(self):
        pass

    """采样器中的样本个数"""
    def __len__(self):
        raise NotImplementedError

    """样本迭代器"""
    def __iter__(self):
        raise NotImplementedError

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

class CPRSampler(Sampler):
    def __init__(self, dataset, sample_ratio, sample_rate, batch_size=1024, n_thread=4, k_interact=None, max_k_interact=3, drop_last=False):
        """
        CPR采样
        :param dataset: Interaction
        :param batch_size:
        :param n_thread:
        """
        super(CPRSampler, self).__init__()
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

    def __iter__(self):
        interact_idx = np.random.choice(self.num_ratings, size=self.choice_size).astype(np.int32)
        if self.cpr_sampler.sample(interact_idx, self.batch_choice_sizes) == -1:
            raise RuntimeError("choice_size is not large enough")
        
        length = len(self.users)
        if self.drop_last:
            n_batch = length // self.batch_size
        else:
            n_batch = ceil(length / self.batch_size)
            
        for i in range(n_batch):
            yield self.users[i * self.batch_sample_size : (i + 1) * self.batch_sample_size], self.items[i * self.batch_sample_size : (i + 1) * self.batch_sample_size]

    def __len__(self):
        length = len(self.users)
        if self.drop_last:
            return length // self.batch_sample_size
        else:
            return (length + self.batch_sample_size - 1) // self.batch_sample_size

class PairwiseSamplerV2(Sampler):
    """采样负样本并构建逐对的训练实例
    训练实例由 batch 组成，每个 batch 包含 batch_user, batch_pos_item 和 batch_neg_item，
    其中 batch_user 和 batch_pos_item 是长度为 batch_size 的列表，batch_neg_item 的长度取决于 num_neg 参数
    参数：
        dataset: Interaction 类型
        num_neg: int, 对于每个正样本采样几个负样本
        batch_size: int
        shuffle: bool
        drop_last: bool
    """
    def __init__(self, dataset, num_neg=1, batch_size=1024, shuffle=True, drop_last=False):
        super(PairwiseSamplerV2, self).__init__()
        if num_neg <= 0:
            raise ValueError("'num_neg' must be a positive integer")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.num_items = dataset.num_items
        self.user_pos_dict = dataset.to_user_dict()  # OrderedDict([int]DataFrame)
        self.num_trainings = sum([len(items) for _, items in self.user_pos_dict.items()])

    def __len__(self):
        if self.drop_last:
            return self.num_trainings // self.batch_size
        return (self.num_trainings - 1 + self.batch_size) // self.batch_size

    def __iter__(self):
        user_arr = np.array(self.user_pos_dict.keys(), dtype=np.int32)
        user_idx = randint_choice(len(user_arr), size=self.num_trainings, replace=True)
        user_list = user_arr[user_idx]

        # 计算每个user在user_list中出现的次数
        user_pos_len = defaultdict(int)
        for u in user_list:
            user_pos_len[u] += 1

        user_pos_sample = dict()
        user_neg_sample = dict()

        # 对每个user采样正样本和负样本
        for user, pos_len in user_pos_len.items():
            try:
                pos_items = self.user_pos_dict[user]  # DataFrame
                pos_idx = randint_choice(len(pos_items), size=pos_len, replace=True)
                pos_idx = pos_idx if isinstance(pos_idx, Iterable) else [pos_idx]
                user_pos_sample[user] = list(pos_items[pos_idx])

                # 从dataset.num_items中删除pos_items就是neg_items的可选值
                neg_items = randint_choice(self.num_items, size=pos_len, replace=True, exclusion=pos_items)
                user_neg_sample[user] = neg_items if isinstance(neg_items, Iterable) else [neg_items]
            except:
                print("Sampling Error")

        pos_item_list = [user_pos_sample[user].pop() for user in user_list]
        neg_item_list = [user_neg_sample[user].pop() for user in user_list]
        # 得到 user_list, pos_item_list, neg_item_list

        data_iter = DataIterator(user_list, pos_item_list, neg_item_list)
        for bat_users, bat_pos_items, bat_neg_items in data_iter:
            yield np.asarray(bat_users), np.asarray(bat_pos_items), np.asarray(bat_neg_items)