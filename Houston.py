import time
from torch.utils.data import DataLoader
import numpy as np
from TOOLS.get_data import Load_my_Dataset
import argparse
import torch
import random
from datetime import datetime
from TOOLS.train import train_clustering

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True

if torch.cuda.is_available():
    device = torch.device("cuda:1")  # 使用GPU
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


# 设置随机数种子
# [7270  860 5390 5191 5734 6265  466 4426 5578 8322]
setup_seed(5191)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scalable ADMM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=2e-6, help='learning rate stage 2')
    parser.add_argument('--n_input', type=int, default=144)
    parser.add_argument("--n_tz", type=int, default=64)
    parser.add_argument("--n_sz", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--rho", type=float, default=3.0, help="Convergence part")
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=10)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--theta", type=float, default=5.0,  help="loss_en")
    parser.add_argument("--eta", type=float, default=0.0003, help="loss_entropy")
    parser.add_argument("--n_cluster", type=int, default=10)
    parser.add_argument("--lamda", type=float, default=0.10)
    parser.add_argument("--dataset", type=str, default="Houston")

    print(datetime.now())

    args = parser.parse_args()
    train_clustering(args,device)



