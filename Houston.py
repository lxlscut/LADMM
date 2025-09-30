import time
import numpy as np
from TOOLS.get_data import Load_my_Dataset
import argparse
import torch
import random
from datetime import datetime
from TOOLS.train import train_clustering

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Set the random seed
# [7270  860 5390 5191 5734 6265  466 4426 5578 8322]
setup_seed(8322)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Scalable ADMM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate stage 2')
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
    parser.add_argument("--eta", type=float, default=0.0000, help="loss_entropy")
    parser.add_argument("--n_cluster", type=int, default=10)
    parser.add_argument("--lamda", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--dataset", type=str, default="Houston")
    print(datetime.now())

    args = parser.parse_args()

    device = torch.device(args.device)  # Use GPU
    print("Using GPU")
    train_clustering(args,device)



