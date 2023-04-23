import os
import torch
import numpy as np
import random
from importlib.util import find_spec
from importlib import import_module
from reckit.configurator import Configurator
from reckit.util.decorators import typeassert

import faulthandler

faulthandler.enable()

def _set_random_seed(seed=2023):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print("set pytorch seed")

@typeassert(recommender=str)
def find_recommender(recommender):
    spec_path = ".".join(["reckit.model", recommender])
    module = None
    
    if find_spec(spec_path):
        module = import_module(spec_path)

    if module is None:
        raise ImportError(f"Recommender: {recommender} not found")
    if hasattr(module, recommender):
        Recommender = getattr(module, recommender)
    else:
        raise ImportError(f"Import {recommender} failed from {module.__file__}!")
    return Recommender

if __name__ == '__main__':
    data_dir = "/home/yjx/projects/GCPR/dataset/"
    root_dir = "/home/yjx/projects/GCPR/"
    config = Configurator(root_dir, data_dir)
    config.add_config(root_dir + "NeuRec.ini", section="NeuRec")
    config.parse_cmd()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
    _set_random_seed(config.seed)
    
    Recommender = find_recommender(config.recommender)

    model_cfg_file = os.path.join(root_dir + "conf", config.recommender + ".ini")
    config.add_config(model_cfg_file, section="hyperparameters")

    model = Recommender(config)
    model.train_model()
