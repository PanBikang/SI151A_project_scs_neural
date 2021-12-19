# Copyright (c) Facebook, Inc. and its affiliates.

#!/usr/bin/env python3

from scs_neural.experimentation import Workspace
import torch
import hydra
import numpy as np
import warnings

warnings.filterwarnings("ignore")

@hydra.main(config_name='scs_neural/configs/matrc.yaml')
def main(cfg):
    cfg = cfg.scs_neural.configs
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    dtype = torch.float64
    torch.set_default_dtype(dtype)

    workspace = Workspace(cfg)
    
    val_loss_avg = workspace.run()
    return val_loss_avg


if __name__ == '__main__':
    main()
