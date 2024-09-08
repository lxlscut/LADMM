import numpy as np
import faiss
import torch
import faiss.contrib.torch_utils

# def find_nearest_neighbors(X, k=20, use_gpu=True):
#     # 数量 of data points (columns in X)
#     num_points = X.shape[1]
#
#     # 转换数据为float32，FAISS要求
#     X = X.astype(np.float32)
#
#     # 创建 FAISS index (默认使用 L2 距离)
#     index = faiss.IndexFlatL2(X.shape[0])
#
#     if use_gpu:
#         # 移动索引到 GPU
#         res = faiss.StandardGpuResources()  # 使用默认资源
#         index = faiss.index_cpu_to_gpu(res, 0, index)  # 0 表示使用第一个 GPU
#
#     # 将数据点加入索引
#     index.add(X.T)  # 转置因为 FAISS 期望每个数据点作为一行
#
#     # 为每个点搜索 k 最近邻
#     _, I = index.search(X.T, k + 1)  # +1 因为点本身是最近邻
#
#     # 初始化关系矩阵为零
#     relationship_matrix = np.zeros((num_points, num_points))
#     # 使用 NumPy 的高级索引填充矩阵
#     # np.arange(num_points)[:, None] 创建列向量为行索引
#     relationship_matrix[np.arange(num_points)[:, None], I[:, 1:]] = 1
#     # return value should be work with column
#     return relationship_matrix.T

def find_nearest_neighbors(X, k=20, use_gpu=True):
    # 数量 of data points (columns in X)
    num_points = X.shape[1]
    # 转换数据为float32，FAISS要求
    X = X.astype(np.float32)
    # 创建 FAISS index (默认使用 L2 距离)
    index = faiss.IndexFlatL2(X.shape[0])
    if use_gpu:
        # 移动索引到 GPU
        res = faiss.StandardGpuResources()  # 使用默认资源
        index = faiss.index_cpu_to_gpu(res, 0, index)  # 0 表示使用第一个 GPU
    # 将数据点加入索引
    index.add(X.T)  # 转置因为 FAISS 期望每个数据点作为一行
    # 为每个点搜索 k 最近邻
    _, I = index.search(X.T, k + 1)  # +1 因为点本身是最近邻
    # 初始化关系矩阵为零
    relationship_matrix = np.zeros((num_points, num_points))
    # 使用 NumPy 的高级索引填充矩阵
    relationship_matrix[np.arange(num_points)[:, None], I[:, 1:]] = 1
    return relationship_matrix.T

def find_nearest_neighbors_tensor(X, k=20, device=None):
    X = torch.nn.functional.normalize(X, p=2, dim=1)
    result = torch.abs(torch.matmul(X,X.T))
    topk_values, topk_indices = torch.topk(result, k, dim=0)
    res = torch.zeros([X.shape[0], X.shape[0]], dtype=torch.float32, device=device)
    res.scatter_(0, topk_indices, 1)
    # res = faiss.StandardGpuResources()
    # gpu_index = faiss.GpuIndexFlatIP(res, X.shape[1])
    # gpu_index = faiss.GpuIndexBinaryFlat
    # gpu_index.add(X)
    # new_d_torch_gpu = torch.empty(X.shape[0], k, device=device, dtype=torch.float32)
    # new_i_torch_gpu = torch.empty(X.shape[0], k, device=device, dtype=torch.int64)
    # gpu_index.search(X, k, new_d_torch_gpu, new_i_torch_gpu)
    # res = torch.zeros([X.shape[0], X.shape[0]], dtype=torch.float32, device=device)
    # row_indices = torch.arange(X.shape[0]).view(-1, 1).expand(X.shape[0], k)
    # res[row_indices, new_i_torch_gpu] = 1
    res.fill_diagonal_(0)
    return res.T