import torch


if __name__ == '__main__':
    a = torch.tensor([[1, 2, 3],[2, 1, 2],[3, 3, 1]])
    args = torch.argsort(a,dim=0,descending=True)

    col_indices = torch.arange(3).unsqueeze(0).expand(3, 3)

    print(args)
    print(col_indices)