
import random
import yaml

import numpy as np
import pytorch_lightning as pl
import torch


from yacs.config import CfgNode as CN


def load_cfg(cfg_file):
    with open(cfg_file, "r") as f:
        yaml_content = yaml.safe_load(f)
    config = CN(yaml_content)
    return config


def set_random_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    random.seed(seed)

    pl.seed_everything(seed, workers=True)
