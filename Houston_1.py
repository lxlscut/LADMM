import time

from torch.utils.data import DataLoader
import numpy as np
from TOOLS.Teacher import L_ADMM
from TOOLS.get_data import Load_my_Dataset
import argparse
import torch
import random

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
setup_seed(42)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scalable ADMM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_input', type=int, default=144)
    parser.add_argument("--n_z", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=7)
    parser.add_argument("--rho", type=float, default=2.0, help="Convergence part")
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--n_cluster", type=int, default=10)
    parser.add_argument("--lamda", type=float, default=0.05)

    args = parser.parse_args()

    dataset = Load_my_Dataset("/home/xianlli/dataset/HSI/Houston/Houston_corrected.mat",
                              "/home/xianlli/dataset/HSI/Houston/Houston_gt.mat", patch_size=args.patch_size,
                              device=device)
    n_cluster = len(torch.unique(dataset.y))
    args.n_cluster = n_cluster
    # args.n_input = 8

    print(args)

    pre_train_loader = DataLoader(dataset, batch_size=4096, shuffle=True)
    model = L_ADMM(args=args).to(device='cuda:0')

    model.pretrain_ae(train_loader=pre_train_loader, num_epochs=50)
    start = time.time()
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True)
    model.train(train_loader=train_loader, dataset=dataset, epochs=600)
    end = time.time()
    print("Elapsed time: {}".format(end - start))
