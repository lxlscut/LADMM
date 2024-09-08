import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from TOOLS.loss_function import combined_entropy_loss, category_error, custom_contrastive_loss
from TOOLS.AE import Ae
import time

from TOOLS.KNN_INI import find_nearest_neighbors_tensor
from TOOLS.early_stopping import EarlyStopping
# 检查CUDA（GPU支持）是否可用
from TOOLS.evaluation import cluster_accuracy
from TOOLS.loss_function import l1_regularization_columnwise, contrastive
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


class classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(classifier, self).__init__()
        # 分类分支
        self.classifier = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=1024),
            # nn.BatchNorm1d(1024),  # 添加批量归一化层
            nn.ReLU(True),
            nn.Linear(in_features=1024, out_features=1024),
            # nn.BatchNorm1d(1024),  # 添加批量归一化层
            nn.ReLU(True),
            nn.Linear(in_features=1024, out_features=512),
            # nn.BatchNorm1d(512),  # 添加批量归一化层
            nn.ReLU(True),
            nn.Linear(512, out_dim),  # Softmax通常不在此处实现，而是在损失函数中处理
            nn.ReLU(True)
        )

    def forward(self, h):
        # reshape data
        h = h.view(h.size(0), -1)
        label_predict = self.classifier(h)
        label_predict = label_predict + 1e-6
        label_predict = F.normalize(input=label_predict, p=1, dim=1)
        return label_predict


class L_ADMM(nn.Module):
    def __init__(self, args):
        super(L_ADMM, self).__init__()
        self.ae = Ae(n_input=args.n_input, n_z=args.n_z)
        self.rho = nn.Parameter(torch.tensor(args.rho))
        self.C_update = C_update()
        self.Z_update = Z_update()
        self.u_update = u_update()
        self.classifier = classifier(in_dim=args.n_z * args.patch_size * args.patch_size, out_dim=args.n_cluster)
        self.softshrink = AdaptiveSoftshrink(lamda=torch.tensor(args.lamda, device=self.rho.device))
        self.num_layer = args.num_layer
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.eta = args.eta
        self.n_cluster = args.n_cluster

    #  pretrain the auto-encoder
    def pretrain_ae(self, train_loader, num_epochs=10, lr=0.001):
        """
        预训练自编码器
        """
        optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr)
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, (data, _, _) in enumerate(train_loader):
                optimizer.zero_grad()
                X_pr, _ = self.ae(data)
                loss = F.mse_loss(input=X_pr, target=data)
                loss.backward()
                optimizer.step()
                total_loss += loss
            print(f'Epoch {epoch + 1}/{num_epochs}, Total Loss: {total_loss}')

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
        # start = time.time()
        m_1 = torch.inverse(m)
        # end = time.time()
        # print(f'Innitialization time: {end - start}')
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
        c_i, y_i, z_s, u_i, h = self.updation(H=H_n, u_0=u_0, z_0=z_0, W_c=W_c, B_c=B_c)
        return c_i, y_i, z_s, u_i, h

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
        label_predict = self.classifier(h)
        c_i, y_i, z_s, u_i, h = self.unfolding_admm(h, z0)
        return X, x_rescon, h, y_i, c_i, z_s, label_predict

    def predict(self, x):
        x_rescon, h = self.ae(x)
        label_predict = self.classifier(h)
        return label_predict

    def predict_batches(self, data_tensor: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        """
        分批次进行预测，针对Tensor输入。

        参数:
        - data_tensor: 输入数据的Tensor，形状为(N, ...)，其中N是样本数，...是每个样本的特征维度。
        - batch_size: 每个批次的样本数量。
        """
        predictions = []  # 用于存储所有批次的预测结果

        # 计算需要多少个批次
        num_batches = (data_tensor.size(0) + batch_size - 1) // batch_size  # 向上取整确保所有数据都被处理

        for i in range(num_batches):
            # 计算当前批次的起始和结束索引
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, data_tensor.size(0))  # 确保不超出数据边界

            # 提取当前批次的数据
            batch_data = data_tensor[start_idx:end_idx]

            # 对当前批次进行预测
            x_rescon, h = self.ae(batch_data)
            label_predict = self.classifier(h)
            predictions.append(label_predict)

        # 将所有批次的预测结果拼接成一个Tensor
        predictions = torch.cat(predictions, dim=0)

        return predictions

    # def refined_subspace_affinity(self, s):
    #     weight = s ** 2 / s.sum(0)
    #     return (weight.t() / weight.sum(1)).t()
    def refined_subspace_affinity(self, s, iter):
        for _ in range(iter):
            weight = s ** 2 / s.sum(0)
            s = (weight.t() / weight.sum(1)).t()
        return s

    def loss(self, X_p, X_r, H, c_i, H_r, label_predict, z_i, epoch, label_true):
        # AE RECONS_LOSS
        loss_ae = F.mse_loss(input=X_r, target=X_p)
        loss_recon = torch.mean(torch.sum(torch.square(H - H_r), dim=0, keepdim=True))
        # loss_recon = custom_contrastive_loss(H=H, H_r=H_r)
        # loss_recon = 1-torch.mean(torch.abs(F.cosine_similarity(H, H_r, dim=0)))
        recons_qulity = torch.norm(H - H_r, p=2, dim=0)
        # loss_en = uniform_loss(label_predict)
        # loss_en = combined_entropy_loss(soft_assignment_matrix=label_predict,alpha=0,beta=1)
        loss_en = category_error(S=c_i, k=label_predict.shape[1], alpha=1)
        # c_loss
        loss_c1 = l1_regularization_columnwise(matrix=c_i, lambda_reg=1)
        # loss_c1 = Elastic_loss(matrix=c_i, lambda_reg=0.5)

        # distill from C to classfication
        loss_c, loss_info = contrastive(
            label_predict, c_i, label_true, recons_qulity)

        s_refine = self.refined_subspace_affinity(label_predict, iter=1)

        loss_class = F.kl_div(label_predict.log(), s_refine.data, reduction='batchmean')
        # loss_class = F.mse_loss(input=label_predict,target=s_refine)
        loss_z = torch.sum(torch.square(c_i - z_i))

        # loss_en = entropy_loss(label_predict)
        loss_balance = combined_entropy_loss(label_predict)

        if epoch >= 200:
            loss = loss_c + self.eta * loss_class

        else:
            loss = loss_ae + self.alpha * loss_recon + self.beta * loss_c1 + self.gamma * loss_z + loss_en

        loss_info.add("loss_ae", loss_ae)
        loss_info.add("loss_recon", loss_recon)
        loss_info.add("loss_c1", loss_c1)
        loss_info.add("loss_c", loss_c)
        loss_info.add("loss_class", loss_class)
        loss_info.add("loss_z", loss_z)
        loss_info.add("loss_en", loss_en)
        loss_info.add("loss", loss)
        loss_info.add("loss_balance", loss_balance)
        return loss, loss_info

    def train(self, train_loader: object, dataset: object, epochs) -> object:
        for epoch in range(epochs):
            if epoch < 200:
                optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
            else:
                # 假设 self.model 是你要冻结的模型
                for param in self.parameters():
                    param.requires_grad = False

                # 确保 self.classifier 的参数需要计算梯度
                for param in self.classifier.parameters():
                    param.requires_grad = True

                # 创建优化器，只传递需要更新的参数
                optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.00001)

            total_loss = 0
            total_loss_c = 0
            total_loss_ae = 0
            total_loss_recon = 0
            total_loss_c1 = 0
            total_loss_class = 0
            total_loss_z = 0
            total_loss_en = 0
            total_loss_balance = 0
            wrong_positive_rate, wrong_negative_rate, positive_sample_number, negative_sample_number = 0, 0, 0, 0
            for batch_idx, (data, label, x_pca) in enumerate(train_loader):
                # initialize Z
                # z0 = find_nearest_neighbors_tensor(X=x_pca, k=10,device=self.rho.device) * 0.1
                z0 = torch.zeros((data.shape[0], data.shape[0]), device=self.rho.device)

                # data = data.to(device)
                optimizer.zero_grad()
                X, x_rescon, h, y_i, c_i, z_s, label_predict = self.forward(data, z0)

                loss, loss_info = self.loss(X_p=X, X_r=x_rescon, H=h, c_i=c_i, H_r=y_i, z_i=z_s,
                                            label_predict=label_predict, label_true=label,
                                            epoch=epoch)
                loss.backward()
                optimizer.step()

                total_loss += loss
                total_loss_c += loss_info.loss_c
                wrong_positive_rate = wrong_positive_rate + loss_info.wrong_positive_rate
                wrong_negative_rate = wrong_negative_rate + loss_info.wrong_negative_rate
                positive_sample_number = positive_sample_number + loss_info.positive_sample_number
                negative_sample_number = negative_sample_number + loss_info.negative_sample_number
                total_loss_ae = total_loss_ae + loss_info.loss_ae
                total_loss_recon = total_loss_recon + loss_info.loss_recon
                total_loss_c1 = total_loss_c1 + loss_info.loss_c1
                total_loss_class = total_loss_class + loss_info.loss_class
                total_loss_z = total_loss_z + loss_info.loss_z
                total_loss_en = total_loss_en + loss_info.loss_en
                total_loss_balance = total_loss_balance + loss_info.loss_balance
            logger.log_variables('Training',
                                 ["loss_ae", total_loss_ae / (batch_idx + 1),
                                  "loss_recon", total_loss_recon / (batch_idx + 1),
                                  "loss_c1", total_loss_c1 / (batch_idx + 1),
                                  "loss_c", total_loss_c / (batch_idx + 1),
                                  "loss_class", total_loss_class / (batch_idx + 1),
                                  "loss", total_loss / (batch_idx + 1),
                                  "loss_z", total_loss_z / (batch_idx + 1),
                                  "wrong_positive_rate", wrong_positive_rate / (batch_idx + 1),
                                  "wrong_negative_rate", wrong_negative_rate / (batch_idx + 1),
                                  "positive_sample_number", positive_sample_number / (batch_idx + 1),
                                  "negative_sample_number", negative_sample_number / (batch_idx + 1),
                                  "loss_en", total_loss_en / (batch_idx + 1),
                                  "loss_balance", total_loss_balance / (batch_idx + 1),
                                  ],
                                 epoch, same_window=True)

            print(
                f'Epoch {epoch + 1}/{400}, Total Loss: {total_loss / (batch_idx + 1), total_loss_c / (batch_idx + 1)}')
            if epoch % 10 == 0 or epoch == epochs - 1:
                acc, nmi, kappa = self.envaluate(x=dataset.train, y=dataset.y)
                print("acc:", acc, "nmi:", nmi, "kappa", kappa)
        logger.close()

    def uniformity_loss(self, encoded_data, t=2.0):
        # 计算余弦相似度
        encoded_norm = F.normalize(encoded_data, p=2, dim=1)
        cosine_sim = torch.matmul(encoded_norm, encoded_norm.T)

        # 对角线元素（自身的相似度）置零
        mask = torch.eye(cosine_sim.size(0), device=cosine_sim.device).bool()
        cosine_sim.masked_fill_(mask, 0)

        # 计算均匀化正则化损失
        loss = torch.logsumexp(-t * cosine_sim, dim=1).mean()

        return loss

    def envaluate(self, x, y):
        self.ae.eval()
        self.classifier.eval()
        with torch.no_grad():
            y = y.cpu().numpy()
            label_predict = self.predict(x)
            # label_predict = self.predict_batches(x,batch_size=4096)
            label_predict = F.softmax(label_predict, dim=1)
        label_predict = label_predict.detach().cpu().numpy()
        label_predict = np.argmax(label_predict, axis=1)

        # number of class and number of data points
        num_class, num_datapoint = np.unique(label_predict, return_counts=True)
        print("num_class", num_class, "num_datapoint", num_datapoint)

        y_best, acc, kappa, nmi, ari, pur, ca = cluster_accuracy(y_true=y, y_pre=label_predict, return_aligned=True)
        return acc, nmi, kappa
