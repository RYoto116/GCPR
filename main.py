import os
import torch
import numpy as np
import random
from importlib.util import find_spec
from importlib import import_module
from reckit.configurator import Configurator
from reckit import typeassert

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
    model_dirs = set(os.listdir("model"))
    model_dirs.remove("base")

    module = None
    for tdir in model_dirs:
        spec_path = ".".join(["model", tdir, recommender])
        if find_spec(spec_path):
            module = import_module(spec_path)
            break

    if module is None:
        raise ImportError(f"Recommender: {recommender} not found")

if __name__ == '__main__':
    data_dir = "/mnt/GCPR/dataset/"
    root_dir = "/mnt/GCPR/"
    configurator = Configurator(root_dir, data_dir)
    configurator.add_config(root_dir + "NeuRec.ini", section="NeuRec")
    configurator.parse_cmd()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(configurator.gpu_id)
    _set_random_seed(configurator.seed)
    Recommender = find_recommender(configurator.recommender)

    model_cfg_file = os.path.join(root_dir + "conf", configurator.recommender + ".ini")
    configurator.add_config(model_cfg_file, section="hyperparameters")

    recommender = Recommender(configurator)
    recommender.train_model()
