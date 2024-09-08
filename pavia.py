from datetime import datetime

import numpy as np
import argparse
import torch
import random

from TOOLS.train import train_clustering

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # 使用GPU
    print("Using GPU")
else:
    device = torch.device("cpu")  # 否则使用CPU
    print("Using CPU")


def setup_seed(seed):
    print("current seed",seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# [7270  860 5390 5191 5734 6265  466 4426 5578 8322]

setup_seed(8322)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scalable ADMM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_input', type=int, default=103)
    parser.add_argument("--n_tz", type=int, default=32)
    parser.add_argument("--n_sz", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=13)
    parser.add_argument("--rho", type=float, default=3.0, help="convergence part")
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=0.0002)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lamda", type=float, default=0.10)
    parser.add_argument("--n_cluster", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="Pavia")

    args = parser.parse_args()
    print(datetime.now())

    train_clustering(args,device)



