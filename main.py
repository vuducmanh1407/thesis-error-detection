#!/usr/bin/python2.7

import torch
from runner import Runner
import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap


parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='debug', type=str)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--vis', action='store_true')
parser.add_argument('--config', type=str)
parser.add_argument('--seed', default=0, type=int)

args = parser.parse_args()

# run = Runner(args)
# run.save_config()

if args.eval:
    # seeds are freezed for visualization, not affecting the results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    run = Runner(args)
    run.save_config()

    colors = np.random.rand(100, 3)
    colors[0] = np.array([0.0, 0.0, 0.0]) # black for error
    colors[1] = np.array([1.0, 1.0, 1.0]) # white for background
    my_cmap = ListedColormap(colors, name="100cmap")
    mpl.colormaps.register(cmap=my_cmap)

    run.model.estimate_thresholds(run.train_loader)
    run.evaluate_new(global_step=-1)
else:
    # fix the seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run = Runner(args)
    run.save_config()
    run.train()