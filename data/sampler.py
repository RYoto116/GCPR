# 采样负样本
import numpy as np
from collections import defaultdict, Iterable
from reckit.rand import randint_choice
from reckit.dataiterator import DataIterator

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
        self.num_neg = num_neg
        self.num_items = dataset.num_items
        self.user_pos_dict = dataset.to_user_dict()  # OrderedDict([int]DataFrame)
        self.num_trainings = sum([len(items) for _, items in self.user_pos_dict.items()])

    def __len__(self):
        if self.drop_last:
            return self.num_trainings // self.batch_size
        return (self.num_trainings - 1 + self.batch_size) // self.batch_size

    def __iter__(self):
        user_arr = np.array(list(self.user_pos_dict.keys()), dtype=np.int32)
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
                pos_items = self.user_pos_dict[user]
                if len(pos_items) == 1:
                    pos_idx = [0] * pos_len
                else:
                    pos_idx = randint_choice(len(pos_items), size=pos_len, replace=True)
                pos_idx = pos_idx if isinstance(pos_idx, Iterable) else [pos_idx]
                user_pos_sample[user] = list(pos_items[pos_idx])

                neg_items = randint_choice(self.num_items, size=pos_len, replace=True, exclusion=self.user_pos_dict[user])
                user_neg_sample[user] = neg_items if isinstance(neg_items, Iterable) else [neg_items]
            except:
                print(user, 'error')

        pos_item_list = [user_pos_sample[user].pop() for user in user_list]
        neg_item_list = [user_neg_sample[user].pop() for user in user_list]
        # 得到 user_list, pos_item_list, neg_item_list

        data_iter = DataIterator(user_list, pos_item_list, neg_item_list, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last)
        for bat_users, bat_pos_items, bat_neg_items in data_iter:
            yield np.asarray(bat_users), np.asarray(bat_pos_items), np.asarray(bat_neg_items)