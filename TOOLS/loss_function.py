import time

import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.signal import correlate

from TOOLS.info import VariableContainer
from TOOLS.sample_generation import generate_negative_sample, robust_sample,generate_positive_samples
# from TOOLS.visdom import FunctionMonitor
import torch

test_m = torch.randn([4096, 4096], device='cuda')


# monitor_p = FunctionMonitor(var_name='ratep', title_name='Random Value Monitoring P')
# monitor_n = FunctionMonitor(var_name='raten', title_name='Random Value Monitoring N')
# monitor_np = FunctionMonitor(var_name='np', title_name='Number of positive samples')
# monitor_nn = FunctionMonitor(var_name='nn', title_name='Number of negative samples')


def preprocess_tensor_for_kl(tensor):
    # 将张量的所有值转换为绝对值
    tensor_abs = torch.abs(tensor)

    # 按列归一化张量，使每列的和为1
    column_sums = tensor_abs.sum(dim=0, keepdim=True)
    tensor_normalized = tensor_abs / column_sums

    return tensor_normalized


def column_min_max_scaling(tensor):
    # 计算每列的最小值和最大值
    min_vals = torch.min(tensor, dim=0, keepdim=True).values
    max_vals = torch.max(tensor, dim=0, keepdim=True).values

    # 对每一列执行缩放
    scaled_tensor = (tensor - min_vals) / (max_vals - min_vals)

    return scaled_tensor


def kl_divergence(input, target):
    # 预处理张量
    p_tensor = preprocess_tensor_for_kl(input)
    q_tensor = preprocess_tensor_for_kl(target)

    # 计算KL散度
    kl_div = F.kl_div(p_tensor.log(), q_tensor, reduction='sum')

    return kl_div


def cosine_similarity_loss_columnwise(x1, x2):
    """
    Computes the cosine similarity loss between two tensors column-wise.

    Args:
    x1 (Tensor): A tensor.
    x2 (Tensor): Another tensor of the same size as x1.

    Returns:
    Tensor: Loss value.
    """
    # Normalize x1 and x2 along the first dimension (column-wise)
    x1_normalized = F.normalize(x1, p=2, dim=0)
    x2_normalized = F.normalize(x2, p=2, dim=0)

    # Compute cosine similarity column-wise
    cos_sim = torch.sum(x1_normalized * x2_normalized, dim=0)
    loss = 1 - cos_sim

    return loss.mean()


def l1_regularization_columnwise(matrix, lambda_reg):
    """
    Apply L1 regularization column-wise to a 2D matrix.

    Args:
    matrix (Tensor): The input 2D tensor (matrix).
    lambda_reg (float): The regularization coefficient.

    Returns:
    Tensor: The L1 regularization loss calculated column-wise.
    """
    return lambda_reg * torch.mean(torch.norm(matrix, p=1, dim=0))

def Elastic_loss(matrix, lambda_reg):
    """
    Apply L1 regularization column-wise to a 2D matrix.

    Args:
    matrix (Tensor): The input 2D tensor (matrix).
    lambda_reg (float): The regularization coefficient.

    Returns:
    Tensor: The L1 regularization loss calculated column-wise.
    """
    return lambda_reg * torch.mean(torch.norm(matrix, p=1, dim=0)) + (1-lambda_reg) * torch.mean(torch.norm(matrix, p=2, dim=0))



def l1_regularization_columnwise_masked(matrix, z, lambda_reg):
    """
    Apply L1 regularization to elements of a 2D matrix ('matrix') where corresponding elements in another
    2D matrix ('z') are zero.

    Args:
    matrix (Tensor): The input 2D tensor (matrix) for which regularization is applied.
    z (Tensor): A 2D tensor of the same shape as 'matrix'. Regularization is applied to elements of 'matrix'
                where corresponding elements in 'z' are zero.
    lambda_reg (float): The regularization coefficient.

    Returns:
    Tensor: The L1 regularization loss calculated for the specified elements.
    """
    assert matrix.shape == z.shape, "The shape of 'matrix' and 'z' must be the same."

    mask = z == 0  # Create a mask where elements of 'z' are zero
    selected_elements = matrix[mask]  # Select elements from 'matrix' using the mask
    # Calculate L1 regularization loss for the selected elements
    return lambda_reg * torch.sum(torch.abs(selected_elements)) / matrix.shape[1]


def l2_regularization_columnwise(matrix, lambda_reg):
    """
    Apply L1 regularization column-wise to a 2D matrix.
    Args:
    matrix (Tensor): The input 2D tensor (matrix).
    lambda_reg (float): The regularization coefficient.

    Returns:
    Tensor: The L1 regularization loss calculated column-wise.
    """
    return lambda_reg * torch.sum(torch.norm(matrix, p=2, dim=0)) / matrix.shape[1]


def custom_loss(V, MASK):
    """
    自定义损失函数，旨在增大MASK矩阵中特定位置值所占的比例。

    参数:
    V - 值矩阵 (tensor)
    MASK - 掩码矩阵，有相同尺寸的二值矩阵 (tensor)

    返回:
    损失值，该值在所需比例增大时减小
    """
    # 确保V和MASK是tensor且具有相同的维度
    assert V.shape == MASK.shape, "V and MASK must have the same shape"

    # 计算sum(V[MASK==1])和sum(V)
    masked_sum = torch.sum(V * MASK)
    total_sum = torch.sum(V)

    # 计算比例
    ratio = masked_sum / total_sum

    # 由于我们想最大化这个比例，我们可以最小化其负值或倒数作为损失
    # 这里我们使用倒数，因为它对小的比例更敏感
    loss = 1 - ratio  # 加上一个小的常数以避免除以零

    return loss


def error_function(C, N, lambda_param):
    """
    计算基于自表示矩阵C和最近邻矩阵N的误差函数。

    参数:
    X -- 数据矩阵 (tensor)
    C -- 自表示矩阵 (tensor)
    N -- 最近邻矩阵 (tensor)
    lambda_param -- 正则化参数 (float)

    返回:
    总误差 (tensor)
    """
    C_norm = F.normalize(input=C, p=2, dim=0)
    # 最近邻正则化
    C_diff = torch.matmul(C_norm.T, C_norm)  # 计算C中每对点的差异
    C_diff.fill_diagonal_(0)
    similarity = N * C_diff

    # loss = 1 - torch.sum(torch.abs(similarity))/(torch.sum(torch.abs(C_diff))+1e-8)

    similarity_mean = torch.sum(similarity) / torch.sum(N)
    loss = 1 - similarity_mean

    return loss


def compute_graph_regularization(Z, A, lambda_reg):
    """
    Compute the graph regularization term using PyTorch.

    Parameters:
    Z (torch.Tensor): The self-representation matrix or feature matrix (size: N x N).
    A (torch.Tensor): The adjacency matrix of the graph (size: N x N).
    lambda_reg (float): The regularization parameter.

    Returns:
    torch.Tensor: The value of the graph regularization term.
    """
    mask = torch.ones(Z.shape[0], Z.shape[0], dtype=torch.bool, device='cuda')
    mask.fill_diagonal_(0)
    N = A.size(0)
    D = torch.diag(A.sum(1))  # Degree matrix
    L = D - A  # Laplacian matrix
    # L_mod = L - torch.diag(torch.diag(L))  # Modified Laplacian with zero diagonal

    reg_term = lambda_reg * torch.trace(torch.matmul(torch.matmul(Z.t(), L), Z))
    return reg_term



def contrastive(label_predict, C, sim, label_true):
    """
    :param label_predict: the output of classifier network
    :param C: the self-representation matrix
    :return:
    """

    loss_info = VariableContainer()
    S = (torch.abs(C) + torch.abs(C.T)) / 2
    S = S.detach()

    Cp, Cn = thrC(S, ro_p=0.90, ro_n=0.10)

    if sim is not None:
        CosP = sim>0.95
        Cp = torch.logical_or(Cp, CosP).to(torch.float32)
        Cn = generate_negative_sample(Cn, Cp)
        CosN = sim<0.5
        Cn = torch.logical_and(Cn, CosN)

    label_true_mask = (label_true[:, None] == label_true).int()

    # 计算 wrong_positive
    wrong_positive = torch.logical_and(Cp, torch.logical_not(label_true_mask))

    # 计算 wrong_positive_rate
    wrong_positive_rate = torch.mean(torch.sum(wrong_positive.float(), dim=0) / (torch.sum(Cp.float(), dim=0) + 1e-8))

    # 计算 positive_sample_number
    positive_sample_number = torch.mean(torch.sum(Cp.float(), dim=0))
    # 计算 wrong_negative
    wrong_negative = torch.logical_and(Cn, label_true_mask)
    # 计算 wrong_negative_rate
    wrong_negative = torch.sum(wrong_negative.float(), dim=0) / (torch.sum(Cn.float(), dim=0) + 1e-8)
    wrong_negative_rate = torch.mean(wrong_negative)
    # 计算 negative_sample_number
    negative_sample_number = torch.mean(torch.sum(Cn.float(), dim=0))


    loss_info.add("positive_sample_number", positive_sample_number)
    loss_info.add("negative_sample_number", negative_sample_number)
    loss_info.add("wrong_negative_rate", wrong_negative_rate)
    loss_info.add("wrong_positive_rate", wrong_positive_rate)

    if label_predict is None:
        return loss_info


    label_predict_norm = F.normalize(label_predict, p=2, dim=1)
    similarity_predict = torch.matmul(label_predict_norm, label_predict_norm.T)

    positive_sum = torch.sum(similarity_predict * Cp, dim=0)
    negative_sum = torch.sum(similarity_predict * Cn+0.01, dim=0)

    temp = (positive_sum + 1e-6) / (positive_sum + negative_sum)
    loss_ = -torch.log(temp)
    loss = torch.mean(loss_)

    return loss, loss_info


def thrC(C, ro_p, ro_n):
    # start = time.time()
    N = C.shape[1]
    C = torch.abs(C)
    Cp = torch.zeros((N, N), device=C.device)
    Cn = torch.zeros((N, N), device=C.device)
    S, Ind = torch.sort(C, dim=0, descending=True)
    # calculate sum of every column
    C_sum = torch.sum(C, dim=0)
    positive_thred = C_sum * ro_p
    negative_thred = C_sum * ro_n
    S_sum = torch.cumsum(S, dim=0)

    S_po = (S_sum <= positive_thred.unsqueeze(0)).to(dtype=torch.int8)
    S_ne = (S_sum >= negative_thred.unsqueeze(0)).to(dtype=torch.int8)

    pos_indices = torch.nonzero(S_po, as_tuple=True)
    neg_indices = torch.nonzero(S_ne, as_tuple=True)

    Cp[Ind[pos_indices[0], pos_indices[1]], pos_indices[1]] = 1
    Cn[Ind[neg_indices[0], neg_indices[1]], neg_indices[1]] = 1
    # end = time.time()
    # Cn_s = get_negative_sample(C, Cp)
    # Cp_sum = torch.sum(Cp, dim=0)
    # Cn_sum = torch.sum(Cn_s, dim=0)
    # times = Cn_sum / Cp_sum
    # Cn = torch.logical_and(Cn, Cn_s)
    # print("where:", end-start, "is_contiguous:", S_po.is_contiguous())

    return Cp, Cn


# Already have Cp, now calculate Cn
def get_negative_sample(C, Cp, times=3):
    """
    :param C: similarity matrix
    :param Cp: positive samples mask
    :param times: how many times negative samples should be extracted
    :return:
    """
    #  get the negative sample number of every data point
    number_postive = torch.sum(Cp, dim=0) * times
    num_points = len(number_postive)
    S, Ind = torch.sort(C, dim=0, descending=False)
    matrix = torch.arange(1, num_points + 1).unsqueeze(1).repeat(1, num_points).to(C.device)
    S_ne = (matrix <= number_postive.unsqueeze(0)).to(dtype=torch.int8)
    neg_indices = torch.nonzero(S_ne, as_tuple=True)
    Cn = torch.zeros((num_points, num_points), device=C.device)
    Cn[Ind[neg_indices[0], neg_indices[1]], neg_indices[1]] = 1
    return Cn



def uniform_loss(x, lambda_=10, epsilon=1e-6):
    # x = x + 1/x.shape[1]
    # x = F.log_softmax(x/0.2, dim=1)
    # x = torch.nn.functional.normalize(x**2,dim=1,p=1)
    pre_mean = torch.mean(x, dim=0, keepdim=True)
    mu = torch.mean(pre_mean)

    # 均匀性误差
    # uniformity_error = torch.sum((pre_mean - mu) ** 2)
    loss_ = torch.abs(pre_mean - mu)
    loss_mean = torch.sum(loss_)

    return loss_mean


def combined_entropy_loss(soft_assignment_matrix, alpha=0.0, beta=1.0):
    num_classes = soft_assignment_matrix.size(1)

    max_column_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))

    # 计算每一行的熵，并取所有行熵的平均值
    row_entropy = -torch.sum(soft_assignment_matrix * torch.log(soft_assignment_matrix + 1e-9), dim=1)
    row_entropy_loss = torch.mean(row_entropy)/max_column_entropy

    # 计算每一列的平均值
    column_means = torch.mean(soft_assignment_matrix, dim=0)

    # 计算这些平均值的熵
    column_entropy = -torch.sum(column_means * torch.log(column_means + 1e-9))

    # 最大值为 log(num_classes)
    normalized_column_entropy = column_entropy / max_column_entropy
    column_loss = 1 - normalized_column_entropy

    # 综合损失，alpha 和 beta 是权重参数
    total_loss = alpha * row_entropy_loss + beta * column_loss

    return total_loss



# 类别误差函数
def category_error(S, k, alpha):
    S = 0.5*(torch.abs(S)+torch.abs(S.T))
    W = torch.abs(S)
    W.fill_diagonal_(0)

    D = torch.diag(W.sum(dim=1))
    L = D - W
    eigvals, _ = torch.linalg.eigh(L)

    eigvals_show = eigvals.detach().cpu().numpy()

    laplacian_term = torch.sum(torch.abs(eigvals[:k]))

    return alpha * laplacian_term



def custom_contrastive_loss(H, H_r):
    # 计算重建和原始矩阵的余弦相似度
    H_norm = F.normalize(H, p=2, dim=1)
    H_r_norm = F.normalize(H_r, p=2, dim=1)

    # 计算余弦相似度矩阵
    cos_sim_matrix = torch.mm(H_norm, H_r_norm.t())

    # cos_sim_matrix_show = cos_sim_matrix.detach().cpu().numpy()

    # 计算对角线元素的和
    diagonal_sum = torch.sum(torch.diag(cos_sim_matrix))

    # 计算整个矩阵元素的和
    total_sum = torch.sum(cos_sim_matrix)

    # 计算比值，作为需要最大化的目标
    ratio = diagonal_sum / total_sum

    # 转换为损失函数形式，1 - ratio 表示我们需要最小化的目标
    loss = 1 - ratio

    return loss


def mutual_information(x, y, bins=30):
    # 将输入张量离散化为直方图
    joint_hist = torch.histc(torch.stack((x, y), dim=1).float(), bins=bins, min=0, max=1)
    joint_hist = joint_hist / joint_hist.sum()

    # 计算联合概率分布
    pxy = joint_hist / torch.sum(joint_hist)

    # 计算边际概率分布
    x_hist = torch.histc(x.float(), bins=bins, min=0, max=1)
    y_hist = torch.histc(y.float(), bins=bins, min=0, max=1)

    px = x_hist / torch.sum(x_hist)
    py = y_hist / torch.sum(y_hist)

    # 计算互信息
    px_py = px.view(-1, 1) * py.view(1, -1)
    nzs = pxy > 0  # 去掉零概率项
    mi = torch.sum(pxy[nzs] * torch.log(pxy[nzs] / px_py[nzs]))

    return mi.item()


def pearson_corrcoef(x, y):
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x - mean_x
    ym = y - mean_y
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))
    r = r_num / r_den
    return r


def compute_negative_entropy_loss(q):
    """
    Compute the negative entropy loss based on the normalized cluster size distribution.

    Parameters:
    q (torch.Tensor): Soft assignment matrix with shape (N, K),
                      where N is the number of samples and K is the number of clusters.

    Returns:
    torch.Tensor: Negative entropy loss.
    """
    # Compute the cluster size distribution Z
    cluster_size = q.sum(dim=0)

    z = cluster_size / cluster_size.sum()

    # Compute entropy H(Z)
    entropy = -torch.sum(z * torch.log(z + 1e-10))  # Adding a small value to avoid log(0)

    # Compute negative entropy loss
    K = q.size(1)
    negative_entropy_loss = torch.log(torch.tensor(K, dtype=torch.float)) - entropy

    # Add regularization term to penalize small values in z
    # regularization = torch.sum(1 / (z + 1e-10))  # Penalize small values
    #
    # # Total loss with regularization
    # loss_with_regularization = negative_entropy_loss + regularization

    return negative_entropy_loss


# def compute_negative_entropy_loss(q):
#     """
#     Compute the weighted negative entropy loss based on the normalized cluster size distribution,
#     with more focus on values approaching zero.
#
#     Parameters:
#     q (torch.Tensor): Soft assignment matrix with shape (N, K),
#                       where N is the number of samples and K is the number of clusters.
#
#     Returns:
#     torch.Tensor: Weighted negative entropy loss.
#     """
#     # Compute the cluster size distribution Z
#     cluster_size = q.sum(dim=0)
#
#     z = cluster_size / cluster_size.sum()
#
#     # Compute weights to emphasize values near zero
#     weights = 1 / (z + 1e-10)  # Adding a small value to avoid division by zero
#     weights /= weights.sum()   # Normalize weights to sum to 1
#
#     # Compute weighted entropy H(Z)
#     weighted_entropy = -torch.sum(weights * z * torch.log(z + 1e-10))
#
#     # Compute weighted negative entropy loss
#     K = q.size(1)
#     weighted_negative_entropy_loss = torch.log(torch.tensor(K, dtype=torch.float)) - weighted_entropy
#
#     return weighted_negative_entropy_loss

def calculate_connected_components(adj_matrix, epsilon=1e-4):
    """
    根据邻接矩阵计算图的联通分量个数
    :param adj_matrix: 邻接矩阵 (PyTorch Tensor)
    :param epsilon: 判定零特征值的阈值
    :return: 联通分量个数
    """
    # 计算度矩阵
    degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))

    # 计算拉普拉斯矩阵
    laplacian_matrix = degree_matrix - adj_matrix

    # 计算特征值
    eigenvalues, _ = torch.linalg.eig(laplacian_matrix)
    # eigenvalues = eigenvalues.real  # 只取实部
    # print("eigenvalues", eigenvalues[:10])
    # 计算接近于零的特征值个数
    num_zero_eigenvalues = torch.sum(torch.abs(eigenvalues) < epsilon).item()

    return num_zero_eigenvalues


import torch


def empty_cluster_penalty_loss(assignment_matrix, lambda_penalty=1.0, epsilon=1e-8):
    """
    仅根据软分配矩阵计算空簇惩罚的损失函数。

    :param assignment_matrix: 软分配矩阵，形状为 (batch_size, num_clusters)
    :param lambda_penalty: 惩罚项的权重
    :param epsilon: 防止除以零的小常数
    :return: 最终损失 (tensor)
    """
    # 计算每个簇的分配总和
    cluster_sums = assignment_matrix.sum(dim=0)

    # 计算空簇惩罚项
    penalty_term = torch.sum(1.0 / (cluster_sums + epsilon))

    # 最终损失仅由空簇惩罚项组成
    total_loss = lambda_penalty * penalty_term
    return total_loss
