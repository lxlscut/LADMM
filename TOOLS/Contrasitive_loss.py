from scipy.sparse.linalg import svds
from torch import nn
import torch
import numpy as np
from TOOLS.info import VariableContainer
from TOOLS.sample_generation import set_top_n_to_one, set_bottom_n_to_one, generate_negative_sample
from sklearn.preprocessing import normalize
from sklearn import cluster
import torch.nn.functional as F


class Contrasitive_loss():
    def __init__(self, n_cluster):
        self.num_p = None
        self.num_n = None
        self.n_cluster = n_cluster
        self.alpha = 0.04

    def thrC(self, C, ro_p):
        # start = time.time()
        N = C.shape[1]
        C = torch.abs(C)
        Cp = torch.zeros((N, N), device=C.device)
        S, Ind = torch.sort(C, dim=0, descending=True)
        # calculate sum of every column
        C_sum = torch.sum(C, dim=0)
        positive_thred = C_sum * ro_p
        S_sum = torch.cumsum(S, dim=0)

        S_po = (S_sum <= positive_thred.unsqueeze(0)).to(dtype=torch.int8)

        pos_indices = torch.nonzero(S_po, as_tuple=True)

        Cp[Ind[pos_indices[0], pos_indices[1]], pos_indices[1]] = C[Ind[pos_indices[0], pos_indices[1]], pos_indices[1]]

        return Cp

    def thrC_MASK(self, C, ro_p, ro_n):
        # start = time.time()
        N = C.shape[1]
        C = torch.abs(C)
        Cp = torch.zeros((N, N), device=C.device)
        Cn = torch.zeros((N, N), device=C.device)
        S, Ind = torch.sort(C, dim=0, descending=True)
        # calculate sum of every column
        C_sum = torch.sum(C, dim=0)
        positive_thred = C_sum * ro_p
        negative_thred = C_sum * (1-ro_n)
        S_sum = torch.cumsum(S, dim=0)

        S_po = (S_sum <= positive_thred.unsqueeze(0)).to(dtype=torch.int8)
        S_ne = (S_sum > negative_thred.unsqueeze(0)).to(dtype=torch.int8)

        pos_indices = torch.nonzero(S_po, as_tuple=True)
        neg_indices = torch.nonzero(S_ne, as_tuple=True)

        Cp[Ind[pos_indices[0], pos_indices[1]], pos_indices[1]] = 1
        Cn[Ind[neg_indices[0], neg_indices[1]], neg_indices[1]] = 1

        return Cp, Cn

    def top_k_to_one(self, C, n):
        """
        Create a column-wise mask for matrix C where the top-n values are set to 1 and the rest to 0.

        Args:
            C (torch.Tensor): input matrix.
            n (int): number of top elements to keep in each column.

        Returns:
            torch.Tensor: mask with ones at the top-n locations of every column.
        """
        # validate the input arguments
        if not isinstance(C, torch.Tensor):
            raise ValueError("Input C must be a torch.Tensor.")
        if n > C.shape[0]:
            raise ValueError("n cannot be greater than the number of rows in C.")

        # obtain the indices of the top-n entries
        _, indices = torch.topk(C, n, dim=0)

        # create an all-zero mask and mark the top-n entries via scatter_
        mask = torch.zeros_like(C, dtype=torch.bool)
        mask.scatter_(0, indices, True)

        return mask

    def estimate_sample_number(self, S):
        # estimate the number of positive and negative points
        # IDEA: apply spectral clustering,usually only the first batch(run once for all other iteration)
        S_thres = self.thrC(S, 0.90)
        S_thres = S_thres.cpu().numpy()
        r = self.n_cluster * 5 + 1
        U, S, _ = svds(S_thres, r, v0=np.ones(S.shape[0]))
        U = U[:, ::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis=1)
        Z = U.dot(U.T)
        Z = Z * (Z > 0)
        L = np.abs(Z ** self.alpha)
        L = L / L.max()
        L = 0.5 * (L + L.T)
        spectral = cluster.SpectralClustering(n_clusters=self.n_cluster, eigen_solver='arpack', affinity='precomputed',
                                              assign_labels='discretize', random_state=42)
        spectral.fit(L)
        grp = spectral.fit_predict(L) + 1
        cluster_index, cluster_count = np.unique(grp, return_counts=True)
        print(np.sum(np.square(cluster_count)) / np.square(np.sum(cluster_count)), cluster_count)
        self.num_p = int(0.8 * np.sum(np.square(cluster_count)))
        self.num_n = int((len(grp) ** 2 - np.sum(np.square(cluster_count))) * 1.0)

    def contrastive(self, label_predict, C, sim, label_true):
        """
        :param label_predict: the output of classifier network
        :param C: the self-representation matrix
        :return:
        """
        loss_info = VariableContainer()
        S = C.detach().clone()  # ensure the original tensor is not modified
        S.fill_diagonal_(0)
        S = F.normalize(S, p=1, dim=0)
        S = (torch.abs(S) + torch.abs(S.T)) / 2
        # if self.num_p == None or self.num_n == None:
        #     self.estimate_sample_number(S)

        Cp, Cn = self.thrC_MASK(C=S, ro_p=0.90, ro_n=0.05)

        if sim is not None:
            CosN = sim < 0.60
            Cn = torch.logical_and(Cn, CosN)


        # ablation study
        # Cp = sim>0.7
        # Cn = sim<0.6


        Cp.fill_diagonal_(False)
        Cn.fill_diagonal_(False)

        Cp_sum = torch.sum(Cp, dim=0, keepdim=True)
        if torch.any(Cp_sum == 0):
            zero_indices = (Cp_sum == 0).nonzero(as_tuple=True)[1]  # Get column indices where sum is 0
            Cp[zero_indices, zero_indices] = 1  # Set the diagonal elements to 1 for these columns

        label_true_mask = (label_true[:, None] == label_true).int()

        # Cp = label_true_mask
        # Cn = torch.logical_not(label_true_mask)

        # compute wrong_positive
        wrong_positive = torch.logical_and(Cp, torch.logical_not(label_true_mask))

        # compute wrong_positive_rate
        wrong_positive_rate = torch.mean(
            torch.sum(wrong_positive.float(), dim=0) / (torch.sum(Cp.float(), dim=0) + 1e-8))

        # compute positive_sample_number
        positive_sample_number = torch.mean(torch.sum(Cp.float(), dim=0))
        # compute wrong_negative
        wrong_negative = torch.logical_and(Cn, label_true_mask)
        # compute wrong_negative_rate
        wrong_negative = torch.sum(wrong_negative.float(), dim=0) / (torch.sum(Cn.float(), dim=0) + 1e-8)
        wrong_negative_rate = torch.mean(wrong_negative)
        # compute negative_sample_number
        negative_sample_number = torch.mean(torch.sum(Cn.float(), dim=0))

        loss_info.add("positive_sample_number", positive_sample_number)
        loss_info.add("negative_sample_number", negative_sample_number)
        loss_info.add("wrong_negative_rate", wrong_negative_rate)
        loss_info.add("wrong_positive_rate", wrong_positive_rate)

        if label_predict is None:
            return loss_info

        if torch.isnan(label_predict).any():
            print("label_predict contains NaN elements. Exiting...")
            exit()  # exit the program

        MASK = torch.logical_or(Cp, Cn)

        label_predict_norm = F.normalize(label_predict, p=2, dim=1)
        similarity_predict = torch.matmul(label_predict_norm, label_predict_norm.T) / 0.2

        max_similarity, _ = torch.max(similarity_predict, dim=1, keepdim=True)
        s_stable = similarity_predict - max_similarity

        sim_exp = torch.exp(s_stable)

        sim_exp_m = sim_exp * MASK.T

        sim_exp_con = s_stable - torch.log(torch.sum(sim_exp_m, dim=1, keepdim=True) + 1e-6)

        pos_sum = torch.sum(sim_exp_con * Cp.T, dim=1)

        sample_numer = torch.sum(Cp.T, dim=1)
        sample_numer[sample_numer == 0] = 1
        if (sample_numer == 0).any():
            # handle edge cases such as avoiding division by zero
            raise ValueError('Division by zero detected in mask_pos_pairs')

        mean_con = pos_sum / (sample_numer)

        loss = -torch.mean(mean_con)

        return loss, loss_info
