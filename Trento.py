import time
from torch.utils.data import DataLoader
import numpy as np
from TOOLS.get_data import Load_my_Dataset
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(10000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scalable ADMM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_input', type=int, default=63)
    parser.add_argument("--n_tz", type=int, default=64)
    parser.add_argument("--n_sz", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--rho", type=float, default=3.0, help="convergence part")
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument("--eta", type=float, default=0.0003)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lamda", type=float, default=0.10)
    parser.add_argument("--n_cluster", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="Trento")


    args = parser.parse_args()

    train_clustering(args,device)



