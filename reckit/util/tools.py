import os
import torch
import torch.nn as nn
from functools import partial
import numpy as np

from reckit import OrderedDict

def inner_product(a, b):
    return torch.sum(a*b, dim=-1)

def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)

def sp_mat_to_sp_tensor(sp_mat):
    coo = sp_mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
    return torch.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()

class InitArg(object):
    MEAN = 0.0
    STDDEV = 0.01
    MIN_VAL = -0.05
    MAX_VAL = 0.05

def truncated_normal_(tensor, mean=0.0, std=1.0):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_(mean=0, std=1)
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

_initializers = OrderedDict()
_initializers["normal"] = partial(nn.init.normal_, mean=InitArg.MEAN, std=InitArg.STDDEV)
_initializers["truncated_normal"] = partial(truncated_normal_, mean=InitArg.MEAN, std=InitArg.STDDEV)
_initializers["uniform"] = partial(nn.init.uniform_, a=InitArg.MIN_VAL, b=InitArg.MAX_VAL)
_initializers["he_normal"] = nn.init.kaiming_normal_
_initializers["he_uniform"] = nn.init.kaiming_uniform_
_initializers["xavier_normal"] = nn.init.xavier_normal_
_initializers["xavier_uniform"] = nn.init.xavier_uniform_
_initializers["zeros"] = nn.init.zeros_
_initializers["ones"] = nn.init.ones_

def get_initializer(init_method):
    if init_method not in _initializers:
        init_list = ', '.join(_initializers.keys())
        raise ValueError(f"'init_method' is invalid, and must be one of '{init_list}'")
    return _initializers[init_method]