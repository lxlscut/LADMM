import numpy as np
import torch
np.random.seed(42)
import torch.nn.functional as F

def calculate(z,k):
    entropy = -torch.sum(z * torch.log(z + 1e-10))  # Adding a small value to avoid log(0)

    # Compute negative entropy loss
    negative_entropy_loss = torch.log(torch.tensor(k, dtype=torch.float)) - entropy
    return negative_entropy_loss




if __name__ == '__main__':
    # a = torch.tensor([0.1,0.6,0.1,0.1,0.1])
    # b = torch.tensor([0.2,0.4,0.2,0.2,0.0])
    # a = torch.log(1000*a+1)
    # b = torch.log(1000*b + 1)
    # res = calculate(b, 5)
    # # [7270  860 5390 5191 5734 6265  466 4426 5578 8322]
    #
    # print(res)
    a = torch.tensor([[1,2,3],[4,5,6],[7,8,9]],dtype=torch.float)
    b = F.normalize(input=a, p=2, dim=1)
    print(b)
