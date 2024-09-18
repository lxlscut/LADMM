import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.modules.module import T
from torch.utils.data import DataLoader, SubsetRandomSampler
from TOOLS.loss_function import combined_entropy_loss, category_error, custom_contrastive_loss
from TOOLS.AE import Ae
import time

from TOOLS.KNN_INI import find_nearest_neighbors_tensor
from TOOLS.early_stopping import EarlyStoppingCurveWithSmooth
# 检查CUDA（GPU支持）是否可用
from TOOLS.evaluation import cluster_accuracy
from TOOLS.loss_function import l1_regularization_columnwise, contrastive
from TOOLS.tools import cos_sim
from TOOLS.visualize import TensorBoardLogger

logger = TensorBoardLogger()


class C_update(nn.Module):
    def __init__(self):
        super(C_update, self).__init__()

    def forward(self, input):
        X, u, Z, rho, W, B = input
        C = torch.matmul(W, X) - torch.matmul(B, u - rho * Z)
        return C


def keep_top_k_values_per_column(input_matrix, k=50):
    values, indices = torch.topk(torch.abs(input_matrix), k, dim=0)  # 沿列维度计算
    threshold = torch.abs(values[k - 1, :])  # 取出每列第k个值作为阈值
    binary_mask = (torch.abs(input_matrix) >= threshold).float()
    return input_matrix * binary_mask, binary_mask


class Z_update(nn.Module):
    def __init__(self):
        super(Z_update, self).__init__()

    def forward(self, input):
        """
        :param input:
        :return: z_s: the sparse value that must reserve
                z_m: the margin value that increase robustness and reduce instability
        """
        rho, C, u = input
        Z = C + u / rho
        Z = Z - torch.diag_embed(torch.diag(Z))
        return Z


class u_update(nn.Module):
    def __init__(self):
        super(u_update, self).__init__()

    def forward(self, input):
        u, rho, C, Z = input
        u = u + rho * (C - Z)
        return u


class AdaptiveSoftshrink(nn.Module):
    def __init__(self, lamda):
        super(AdaptiveSoftshrink, self).__init__()
        self.relu = nn.ReLU()
        self.lamda = lamda

    def forward(self, x, rho):
        # 计算绝对值减去lambda后的结果，并应用ReLU
        shrunk = self.relu(torch.abs(x) - self.lamda / rho)
        # shrunk = self.relu(torch.abs(x) - 0.005)

        # 恢复原始值的符号
        z_s = shrunk * torch.sign(x)
        return z_s


class admm_norm(nn.Module):
    def __init__(self):
        super(admm_norm, self).__init__()

    def forward(self, x):
        x = x + 1e-8  # 防止除零
        norm = torch.norm(x, p=1, dim=-1, keepdim=True)  # L1 范数
        return x / norm




class Teacher(nn.Module):
    def __init__(self, args):
        super(Teacher, self).__init__()
        self.ae = Ae(n_input=args.n_input, n_z=args.n_tz)
        self.rho = nn.Parameter(torch.tensor(args.rho))
        self.C_update = C_update()
        self.Z_update = Z_update()
        self.u_update = u_update()
        self.softshrink = AdaptiveSoftshrink(lamda=torch.tensor(args.lamda, device=self.rho.device))
        self.num_layer = args.num_layer
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.theta = args.theta
        self.eta = args.eta
        self.n_cluster = args.n_cluster
        # first stage:
        self.stopping_1 = EarlyStoppingCurveWithSmooth(min_delta=1e-3, patience=10)
        # second stage:


    #  pretrain the auto-encoder
    def pretrain_ae(self, train_loader, path, trainable=True, num_epochs=10, lr=0.001):
        """
        预训练自编码器
        """

        self.ae.initialize_weights()

        # 确保 path 参数不为 None
        assert path is not None, "The path to save/load weights must be provided."

        if trainable:
            optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr)
            for epoch in range(num_epochs):
                total_loss = 0
                for batch_idx, (data, _, _) in enumerate(train_loader):
                    optimizer.zero_grad()
                    X_pr, _ = self.ae(data)
                    loss = F.mse_loss(input=X_pr, target=data)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f'Epoch {epoch}/{num_epochs}, Total Loss: {(total_loss)/(batch_idx+1)}')

            # Save the trained weights to the specified path
            torch.save(self.ae.state_dict(), path)
            print(f'Saved weights to {path}')

        # Load the weights from the specified path
        self.ae.load_state_dict(torch.load(path))
        print(f'Loaded weights from {path}')



    def updation(self, H, u_0, z_0, W_c, B_c):
        u = u_0
        Z = z_0
        X = F.normalize(H, p=2, dim=0)
        for i in range(self.num_layer):
            c_i = self.C_update([X, u, Z, self.rho, W_c, B_c])
            z_i = self.Z_update([self.rho, c_i, u])
            z_s = self.softshrink(z_i, self.rho)
            u_i = self.u_update([u, self.rho, c_i, z_s])
            u = u_i
            Z = z_s
        y_i = torch.matmul(X, c_i)
        return c_i, y_i, z_s, u_i, X

    def innitialize_c(self, X, rho):
        m = 2 * torch.matmul(X.T, X) + rho * torch.eye(X.shape[1], device=self.rho.device)
        m_1 = torch.inverse(m)
        W_c = torch.matmul(m_1, 2 * X.T)
        B_c = m_1
        return W_c, B_c

    def unfolding_admm(self, H, z_0):
        # initialize variable
        u_0 = torch.ones((H.shape[0], H.shape[0]), device=self.rho.device) * 1e-3
        # initialize W, B
        H_ = H.view(H.shape[0], -1).T
        H_n = F.normalize(input=H_, p=2, dim=0)
        W_c, B_c = self.innitialize_c(H_n, self.rho)
        c_i, h_r, z_s, u_i, h = self.updation(H=H_n, u_0=u_0, z_0=z_0, W_c=W_c, B_c=B_c)
        return c_i, h_r, z_s, u_i, h

    def forward(self, X, z0):
        """
        :param X: HSI data
        :return: X: HSI data;
                 x_rescon: reconstruction data from AE
                 h: the latent code
                 y_i: the reconstruction value of latent code by self-expression property
                 c_i: the self-representation matrix
                 z_i: the Auxiliary variables to help achieve sparse
        """
        x_rescon, h = self.ae(X)
        c_i, h_r, z_s, u_i, h = self.unfolding_admm(h, z0)
        return X, x_rescon, h, h_r, c_i, z_s

    def loss(self, X_p, X_r, H, c_i, H_r, sim, z_i, n_cluster, label_true):

        # AE RECONS_LOSS
        loss_ae = F.mse_loss(input=X_r, target=X_p)
        loss_recon = torch.mean(torch.sum(torch.square(H - H_r), dim=0, keepdim=True))
        # c_loss
        loss_c1 = l1_regularization_columnwise(matrix=c_i, lambda_reg=1)

        # distill from C to classfication
        loss_info = contrastive(None, c_i, sim, label_true)

        loss_z = torch.sum(torch.square(c_i - z_i))

        loss_en = category_error(S=c_i, k=n_cluster, alpha=1)

        # loss = loss_ae + self.alpha * loss_recon + self.beta * loss_c1 + self.gamma * loss_z + loss_en
        loss = loss_ae + self.alpha * loss_recon + self.beta * loss_c1 + self.gamma * loss_z + self.theta*loss_en
        # print("loss_ae", loss_ae)

        loss_info.add("loss_ae", loss_ae)
        loss_info.add("loss_recon", loss_recon)
        loss_info.add("loss_c1", loss_c1)
        loss_info.add("loss_z", loss_z)
        loss_info.add("loss", loss)
        loss_info.add("loss_en", loss_en)
        return loss, loss_info

    def train_stage_1(self, train_loader, dataset, epochs):
        self.ae.initialize_weights()
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.001,
                                                         min_lr=1e-10, eps=1e-10, verbose=True)
        smoothed_total_loss = None
        for epoch in range(epochs):
            total_loss = 0
            total_loss_ae = 0
            total_loss_recon = 0
            total_loss_c1 = 0
            total_loss_z = 0
            total_loss_en = 0
            wrong_positive_rate, wrong_negative_rate, positive_sample_number, negative_sample_number = 0, 0, 0, 0
            for batch_idx, (data, label, x_pca) in enumerate(train_loader):
                optimizer.zero_grad()
                similarity_matrix = cos_sim(data)
                z0 = torch.zeros((data.shape[0], data.shape[0]), device=self.rho.device)
                X, x_rescon, h, y_i, c_i, z_s = self.forward(data, z0)

                loss, loss_info = self.loss(X_p=X, X_r=x_rescon, H=h, c_i=c_i, H_r=y_i, z_i=z_s, sim = similarity_matrix,
                                            n_cluster=self.n_cluster, label_true=label)
                loss.backward()
                optimizer.step()

                total_loss += loss
                wrong_positive_rate = wrong_positive_rate + loss_info.wrong_positive_rate
                wrong_negative_rate = wrong_negative_rate + loss_info.wrong_negative_rate
                positive_sample_number = positive_sample_number + loss_info.positive_sample_number
                negative_sample_number = negative_sample_number + loss_info.negative_sample_number
                total_loss_ae = total_loss_ae + loss_info.loss_ae
                total_loss_recon = total_loss_recon + loss_info.loss_recon
                total_loss_c1 = total_loss_c1 + loss_info.loss_c1
                total_loss_en = total_loss_en + loss_info.loss_en
                total_loss_z = total_loss_z + loss_info.loss_z
            if smoothed_total_loss == None:
                smoothed_total_loss = total_loss
            smooth_total_loss = 0.5 * smoothed_total_loss + (1 - 0.5) * total_loss
            scheduler.step(smooth_total_loss/(batch_idx+1))
            print("current learning rate:", optimizer.param_groups[0]['lr'])
            if self.stopping_1.early_stop==False:
                self.stopping_1(total_loss / (batch_idx + 1))
                if self.stopping_1.early_stop == True:
                    torch.save(self.state_dict(), 'checkpoint/teacher_checkpoint.pt')
                    # self.load_state_dict(torch.load('checkpoint/teacher_checkpoint.pt'))
                    break
            if epoch==epochs-1:
                torch.save(self.state_dict(), 'checkpoint/teacher_checkpoint.pt')
                # self.load_state_dict(torch.load('checkpoint/teacher_checkpoint.pt'))
                break
            logger.log_variables('Training',
                                 ["loss_ae", total_loss_ae / (batch_idx + 1),
                                  "loss_recon", total_loss_recon / (batch_idx + 1),
                                  "loss_c1", total_loss_c1 / (batch_idx + 1),
                                  "loss", total_loss / (batch_idx + 1),
                                  "loss_z", total_loss_z / (batch_idx + 1),
                                  "loss_en", total_loss_en / (batch_idx + 1),
                                  "wrong_positive_rate", wrong_positive_rate / (batch_idx + 1),
                                  "wrong_negative_rate", wrong_negative_rate / (batch_idx + 1),
                                  "positive_sample_number", positive_sample_number / (batch_idx + 1),
                                  "negative_sample_number", negative_sample_number / (batch_idx + 1),
                                  ],
                                 epoch, same_window=True)
            print(f'Epoch {epoch + 1}/{400}, Total Loss: {total_loss / (batch_idx + 1)}')
