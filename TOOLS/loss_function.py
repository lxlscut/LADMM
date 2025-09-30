import time

import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy.signal import correlate
from sklearn.cluster import SpectralClustering
from TOOLS.info import VariableContainer
from TOOLS.sample_generation import generate_negative_sample, robust_sample, generate_positive_samples, \
    set_top_n_to_one, set_bottom_n_to_one
# from TOOLS.visdom import FunctionMonitor
import torch

test_m = torch.randn([4096, 4096], device='cuda')



# # NOTE: currently unused function
# def preprocess_tensor_for_kl(tensor):
#     # convert all tensor values to their absolute value
#     tensor_abs = torch.abs(tensor)

#     # normalize each column so that it sums to one
#     column_sums = tensor_abs.sum(dim=0, keepdim=True)
#     tensor_normalized = tensor_abs / column_sums

#     return tensor_normalized


# # NOTE: currently unused function
# def column_min_max_scaling(tensor):
#     # compute the column-wise minima and maxima
#     min_vals = torch.min(tensor, dim=0, keepdim=True).values
#     max_vals = torch.max(tensor, dim=0, keepdim=True).values

#     # scale each column individually
#     scaled_tensor = (tensor - min_vals) / (max_vals - min_vals)

#     return scaled_tensor


# # NOTE: currently unused function
# def kl_divergence(input, target):
#     # preprocess tensors
#     p_tensor = preprocess_tensor_for_kl(input)
#     q_tensor = preprocess_tensor_for_kl(target)

#     # compute the KL divergence
#     kl_div = F.kl_div(p_tensor.log(), q_tensor, reduction='sum')

#     return kl_div


# # NOTE: currently unused function
# def cosine_similarity_loss_columnwise(x1, x2):
#     """
#     Computes the cosine similarity loss between two tensors column-wise.

#     Args:
#     x1 (Tensor): A tensor.
#     x2 (Tensor): Another tensor of the same size as x1.

#     Returns:
#     Tensor: Loss value.
#     """
#     # normalize x1 and x2 along the first dimension (column-wise)
#     x1_normalized = F.normalize(x1, p=2, dim=0)
#     x2_normalized = F.normalize(x2, p=2, dim=0)

#     # compute cosine similarity column-wise
#     cos_sim = torch.sum(x1_normalized * x2_normalized, dim=0)
#     loss = 1 - cos_sim

#     return loss.mean()


def l1_regularization_columnwise(matrix):
    """
    Apply L1 regularization column-wise to a 2D matrix.

    Args:
    matrix (Tensor): The input 2D tensor (matrix).
    lambda_reg (float): The regularization coefficient.
    Returns:
    Tensor: The L1 regularization loss calculated column-wise.
    """
    col = torch.norm(matrix, p=1, dim=0)
    col_mean = col.mean()
    return col_mean

# # NOTE: currently unused function
# def contrastive(label_predict, C, sim, label_true):
#     """
#     :param label_predict: the output of classifier network
#     :param C: the self-representation matrix
#     :return:
#     """
#     loss_info = VariableContainer()
#     S = (torch.abs(C) + torch.abs(C.T)) / 2
#     S = S.detach()
#     num_points = len(label_true)
#     num_p = 0.8*(num_points*num_points)/9
#     num_n = 0.8*(num_points*num_points)*(8/9)
#     Cp = set_top_n_to_one(S, num_p)
#     Cn = set_bottom_n_to_one(S, num_n)
#     if sim is not None:
#         CosP = sim > 0.95
#         Cp = torch.logical_or(Cp, CosP).to(torch.float32)
#         CosN = sim < 0.20
#         Cn = torch.logical_and(Cn, CosN)




#     Cp.fill_diagonal_(False)
#     Cn.fill_diagonal_(False)

#     label_true_mask = (label_true[:, None] == label_true).int()

#     # compute wrong_positive
#     wrong_positive = torch.logical_and(Cp, torch.logical_not(label_true_mask))

#     # compute wrong_positive_rate
#     wrong_positive_rate = torch.mean(torch.sum(wrong_positive.float(), dim=0) / (torch.sum(Cp.float(), dim=0) + 1e-8))

#     # compute positive_sample_number
#     positive_sample_number = torch.mean(torch.sum(Cp.float(), dim=0))
#     # compute wrong_negative
#     wrong_negative = torch.logical_and(Cn, label_true_mask)
#     # compute wrong_negative_rate
#     wrong_negative = torch.sum(wrong_negative.float(), dim=0) / (torch.sum(Cn.float(), dim=0) + 1e-8)
#     wrong_negative_rate = torch.mean(wrong_negative)
#     # compute negative_sample_number
#     negative_sample_number = torch.mean(torch.sum(Cn.float(), dim=0))

#     loss_info.add("positive_sample_number", positive_sample_number)
#     loss_info.add("negative_sample_number", negative_sample_number)
#     loss_info.add("wrong_negative_rate", wrong_negative_rate)
#     loss_info.add("wrong_positive_rate", wrong_positive_rate)

#     if label_predict is None:
#         return loss_info

#     label_predict_norm = F.normalize(label_predict, p=2, dim=1)
#     similarity_predict = torch.matmul(label_predict_norm, label_predict_norm.T)

#     pos_sim = similarity_predict*Cp
#     pos_show = pos_sim.detach().cpu().numpy()


#     positive_sum_r = torch.sum(similarity_predict * Cp.T, dim=1)
#     negative_sum_r = torch.sum(similarity_predict * Cn.T, dim=1)

#     positive_sum_c = torch.sum(similarity_predict * Cp, dim=0)
#     negative_sum_c = torch.sum(similarity_predict * Cn, dim=0)

#     positive_sum = positive_sum_r + positive_sum_c
#     negative_sum = negative_sum_r + negative_sum_c

#     temp = (positive_sum + 1e-6) / (positive_sum + negative_sum + 2)
#     loss_ = -torch.log(temp/2)
#     loss = torch.mean(loss_)


#     return loss, loss_info


# # NOTE: currently unused function
# def thrC(C, ro_p, ro_n):
#     # start = time.time()
#     N = C.shape[1]
#     C = torch.abs(C)
#     Cp = torch.zeros((N, N), device=C.device)
#     Cn = torch.zeros((N, N), device=C.device)
#     S, Ind = torch.sort(C, dim=0, descending=True)
#     # calculate sum of every column
#     C_sum = torch.sum(C, dim=0)
#     positive_thred = C_sum * ro_p
#     negative_thred = C_sum * ro_n
#     S_sum = torch.cumsum(S, dim=0)

#     S_po = (S_sum <= positive_thred.unsqueeze(0)).to(dtype=torch.int8)
#     S_ne = (S_sum >= negative_thred.unsqueeze(0)).to(dtype=torch.int8)

#     pos_indices = torch.nonzero(S_po, as_tuple=True)
#     neg_indices = torch.nonzero(S_ne, as_tuple=True)

#     Cp[Ind[pos_indices[0], pos_indices[1]], pos_indices[1]] = 1
#     Cn[Ind[neg_indices[0], neg_indices[1]], neg_indices[1]] = 1

#     return Cp, Cn


# # NOTE: currently unused function
# def topk_rows_and_columns_to_one(matrix, k):
#     # mark the top-k entries per row and per column using boolean masks for efficiency

#     # get the row-wise top-k indices and build the row mask
#     row_values, row_indices = torch.topk(matrix, k, dim=1)
#     row_mask = torch.zeros_like(matrix, dtype=torch.bool).scatter_(1, row_indices, True)

#     # get the column-wise top-k indices and build the column mask
#     col_values, col_indices = torch.topk(matrix, k, dim=0)
#     col_mask = torch.zeros_like(matrix, dtype=torch.bool).scatter_(0, col_indices, True)

#     # take the union of row and column masks
#     final_mask = row_mask | col_mask

#     return final_mask


# # Already have Cp, now calculate Cn
# # NOTE: currently unused function
# def get_negative_sample(C, Cp, times=3):
#     """
#     :param C: similarity matrix
#     :param Cp: positive samples mask
#     :param times: how many times negative samples should be extracted
#     :return:
#     """
#     #  get the negative sample number of every data point
#     number_postive = torch.sum(Cp, dim=0) * times
#     num_points = len(number_postive)
#     S, Ind = torch.sort(C, dim=0, descending=False)
#     matrix = torch.arange(1, num_points + 1).unsqueeze(1).repeat(1, num_points).to(C.device)
#     S_ne = (matrix <= number_postive.unsqueeze(0)).to(dtype=torch.int8)
#     neg_indices = torch.nonzero(S_ne, as_tuple=True)
#     Cn = torch.zeros((num_points, num_points), device=C.device)
#     Cn[Ind[neg_indices[0], neg_indices[1]], neg_indices[1]] = 1

#     return Cn


# # NOTE: currently unused function
# def uniform_loss(x, lambda_=10, epsilon=1e-6):
#     # x = x + 1/x.shape[1]
#     # x = F.log_softmax(x/0.2, dim=1)
#     # x = torch.nn.functional.normalize(x**2,dim=1,p=1)
#     pre_mean = torch.mean(x, dim=0, keepdim=True)
#     mu = torch.mean(pre_mean)

#     # uniformity error
#     # uniformity_error = torch.sum((pre_mean - mu) ** 2)
#     loss_ = torch.abs(pre_mean - mu)
#     loss_mean = torch.sum(loss_)

#     return loss_mean


# # NOTE: currently unused function
# def combined_entropy_loss(soft_assignment_matrix, alpha=0.0, beta=1.0):
#     num_classes = soft_assignment_matrix.size(1)

#     max_column_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))

#     # compute the entropy of each row and average them
#     row_entropy = -torch.sum(soft_assignment_matrix * torch.log(soft_assignment_matrix + 1e-9), dim=1)
#     row_entropy_loss = torch.mean(row_entropy) / max_column_entropy

#     # compute the mean of every column
#     column_means = torch.mean(soft_assignment_matrix, dim=0)

#     # measure the entropy of these means
#     column_entropy = -torch.sum(column_means * torch.log(column_means + 1e-9))

#     # the maximum entropy equals log(num_classes)
#     normalized_column_entropy = column_entropy / max_column_entropy
#     column_loss = 1 - normalized_column_entropy

#     # combine losses with weights alpha and beta
#     total_loss = alpha * row_entropy_loss + beta * column_loss

#     return total_loss


# category error function
def category_error(S, k, alpha):
    S = 0.5 * (torch.abs(S) + torch.abs(S.T))

    W = torch.abs(S)
    W.fill_diagonal_(0)

    D = torch.diag(W.sum(dim=1))
    L = D - W
    eigvals, ei_vector = torch.linalg.eigh(L)

    laplacian_term = torch.sum(torch.abs(eigvals[:k]))

    return alpha * laplacian_term


# # NOTE: currently unused function
# def custom_contrastive_loss(H, H_r):
#     # compute the cosine similarity between the reconstructed and original matrices
#     H_norm = F.normalize(H, p=2, dim=1)
#     H_r_norm = F.normalize(H_r, p=2, dim=1)

#     # compute the cosine similarity matrix
#     cos_sim_matrix = torch.mm(H_norm, H_r_norm.t())

#     # cos_sim_matrix_show = cos_sim_matrix.detach().cpu().numpy()

#     # sum the diagonal elements
#     diagonal_sum = torch.sum(torch.diag(cos_sim_matrix))

#     # sum all elements in the matrix
#     total_sum = torch.sum(cos_sim_matrix)

#     # compute the ratio to maximise
#     ratio = diagonal_sum / total_sum

#     # convert to a loss by minimising 1 - ratio
#     loss = 1 - ratio

#     return loss


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
    return negative_entropy_loss


# # NOTE: currently unused function
# def calculate_connected_components(adj_matrix, epsilon=1e-4):
#     """
#     compute the number of connected components from an adjacency matrix
#     :param adj_matrix: adjacency matrix (PyTorch Tensor)
#     :param epsilon: threshold to determine zero eigenvalues
#     :return: number of connected components
#     """
#     # compute the degree matrix
#     degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))

#     # compute the Laplacian matrix
#     laplacian_matrix = degree_matrix - adj_matrix

#     # compute eigenvalues
#     eigenvalues, _ = torch.linalg.eig(laplacian_matrix)
#     # eigenvalues = eigenvalues.real  # take the real part only
#     # print("eigenvalues", eigenvalues[:10])
#     # count eigenvalues that are close to zero
#     num_zero_eigenvalues = torch.sum(torch.abs(eigenvalues) < epsilon).item()

#     return num_zero_eigenvalues


def empty_cluster_penalty_loss(assignment_matrix, lambda_penalty=1.0, epsilon=1e-8):
    """
    Compute only the empty-cluster penalty from the soft assignment matrix.

    :param assignment_matrix: soft assignment matrix of shape (batch_size, num_clusters)
    :param lambda_penalty: penalty weight
    :param epsilon: small constant to avoid division by zero
    :return: resulting penalty (tensor)
    """
    # compute the total assignment weight for each cluster
    cluster_sums = assignment_matrix.sum(dim=0)

    # compute the empty-cluster penalty
    penalty_term = torch.sum(1.0 / (cluster_sums + epsilon))

    # the loss is composed solely of the penalty term
    total_loss = lambda_penalty * penalty_term
    return total_loss


# # NOTE: currently unused function
# def compute_clustering_matrix(similarity_matrix, n_clusters):
#     """
#     compute a clustering matrix from a similarity matrix.

#     Args:
#     - similarity_matrix: similarity matrix as a torch.Tensor with shape (n_samples, n_samples)
#     - n_clusters: number of clusters

#     Returns:
#     - clustering_matrix: clustering matrix as a torch.Tensor with shape (n_samples, n_samples)
#     """
#     # convert the tensor to numpy for scikit-learn
#     similarity_matrix_np = similarity_matrix.cpu().numpy()

#     # apply spectral clustering
#     clustering = SpectralClustering(
#         n_clusters=n_clusters,
#         affinity='precomputed',
#         assign_labels='discretize',
#         random_state=0
#     )
#     cluster_labels = clustering.fit_predict(similarity_matrix_np)

#     # convert cluster labels back to a torch.Tensor
#     cluster_labels_tensor = torch.from_numpy(cluster_labels).to(similarity_matrix.device)

#     # build the clustering matrix with broadcasting to avoid loops
#     # clustering_matrix[i, j] = 1 if cluster_labels[i] == cluster_labels[j]
#     clustering_matrix = (cluster_labels_tensor.unsqueeze(1) == cluster_labels_tensor.unsqueeze(0)).int()

#     return clustering_matrix