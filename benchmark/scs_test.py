# Copyright (c) Facebook, Inc. and its affiliates.

#!/usr/bin/env python3

from scs_neural.experimentation import Workspace
import torch
import hydra
import numpy as np
import warnings
import os, sys

warnings.filterwarnings("ignore")

@hydra.main(config_name='scs_neural/configs/lasso.yaml')
def main(cfg):

    fname = os.getcwd()[:-8] + '18-08-25' + '/latest.pt'
    print(fname)
    if os.path.exists(fname):
        print(f'Resuming fom {fname}')
        with open(fname, 'rb') as f:
            workspace = torch.load(f, map_location='cpu')

    workspace.test()
if __name__ == '__main__':
    main()
