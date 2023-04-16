import numpy as np
from reckit.cython import CyCPRSampler

class _Dataset(object):
    """
    数据集，包括 user_list, pos_item_list 和 neg_item_list
    """
    def __init__(self, data):
        for d in data:
            if len(d) != len(data[0]):
                raise ValueError("The length of the given data are not equal!")
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return [data[idx] for data in self.data]

class Sampler(object):
    """Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class RandomSampler(Sampler):
    """
    随机采样元素，不放回
    arg:
        data_source (_Dataset): Dataset to sample from.
    """
    def __init__(self, data_source):
        super(RandomSampler, self).__init__()
        self.data_source = data_source

    def __iter__(self):
        perm = np.random.permutation(len(self.data_source)).tolist()
        return iter(perm)

    def __len__(self):
        return len(self.data_source)

class SequentialSampler(Sampler):
    def __init__(self, data_source):
        super(SequentialSampler, self).__init__()
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


class BatchSampler(Sampler):
    """Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler)
        batch_size (int)
        drop_last (bool)
    """
    def __init__(self, sampler, batch_size, drop_last):
        super(BatchSampler, self).__init__()
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size

class _DataLoaderIter(object):
    """用sampler指定的方式遍历一遍数据集
    Args:
        loader: (DataIterator)
    """
    def __init__(self, loader):
        self.dataset = loader.dataset
        self.batch_sampler = loader.batch_sampler
        self.sample_iter = iter(self.batch_sampler)

    def __next__(self):
        indices = next(self.sample_iter)  # [batch_size]
        batch = [self.dataset[i] for i in indices]
        transposed = [list(sample) for sample in zip(*batch)]  # 转置，[item * user]
        if len(transposed) == 1:
            transposed = transposed[0]
        return transposed

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        return self

class DataIterator(object):
    def __init__(self, *data, batch_size=1, shuffle=False, drop_last=False):
        self.dataset = _Dataset(list(data))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        if shuffle:
            sampler = RandomSampler(self.dataset)
        else:
            sampler = SequentialSampler(self.dataset)
        self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)

    def __iter__(self):
        return _DataLoaderIter(self)

    def __len__(self):
        return len(self.batch_sampler)