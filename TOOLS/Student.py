from torch.optim import lr_scheduler

from TOOLS.AE import Ae
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from TOOLS.early_stopping import EarlyStoppingCurveWithSmooth
from TOOLS.evaluation import cluster_accuracy
from TOOLS.loss_function import contrastive, compute_negative_entropy_loss, empty_cluster_penalty_loss
from TOOLS.tools import cos_sim
from TOOLS.visualize import TensorBoardLogger

logger = TensorBoardLogger()


class classifier(nn.Module):
    def __init__(self, in_dim, out_dim, patch_size):
        super(classifier, self).__init__()
        # 分类分支
        self.encoder = nn.Sequential()
        self.encoder.add_module("cov01", nn.Conv2d(in_channels=in_dim, out_channels=128, kernel_size=[3, 3], stride=1,
                                                   padding='same'))
        self.encoder.add_module("bn01", nn.BatchNorm2d(128))
        self.encoder.add_module("relu01", nn.LeakyReLU(negative_slope=0.01))

        # self.encoder.add_module("cov02", nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[3, 3], stride=1,
        #                                            padding='same'))
        # self.encoder.add_module("bn02", nn.BatchNorm2d(64))
        # self.encoder.add_module("relu02", nn.LeakyReLU(negative_slope=0.01))


        self.encoder.add_module("cov03", nn.Conv2d(in_channels=128, out_channels=32, kernel_size=[3, 3], stride=1,
                                                   padding='same'))
        self.encoder.add_module("bn03", nn.BatchNorm2d(32))
        self.encoder.add_module("relu03", nn.LeakyReLU(negative_slope=0.01))

        # self.encoder.add_module("cov04", nn.Conv2d(in_channels=32, out_channels=16, kernel_size=[3, 3], stride=1,
        #                                            padding='same'))
        # self.encoder.add_module("bn04", nn.BatchNorm2d(16))
        # self.encoder.add_module("relu04", nn.ReLU(True))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=32*patch_size*patch_size, out_features=1024),
            nn.BatchNorm1d(1024),  # 添加批量归一化层
            nn.ReLU(True),
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),  # 添加批量归一化层
            nn.ReLU(True),
            nn.Linear(in_features=512, out_features=out_dim),
            nn.ReLU(True),
            # nn.Softmax()
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # reshape data
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        label_predict = self.classifier(h)
        # soft_assignments = F.softmax(label_predict, dim=1)

        label_predict = label_predict + 1e-6
        label_predict = F.normalize(input=label_predict, p=1, dim=1)

        # label_predict = F.softmax(label_predict, dim=1)

        return label_predict


class Student(nn.Module):
    def __init__(self, args):
        super(Student, self).__init__()
        self.classifier = classifier(in_dim=args.n_input, out_dim=args.n_cluster,patch_size=args.patch_size)
        self.stopping_2 = EarlyStoppingCurveWithSmooth(min_delta=1e-5, patience=30)
        self.eta = args.eta
    def forward(self, input):
        # _, H = self.ae(input)
        output = self.classifier(input)
        return output

    def refined_subspace_affinity(self, s):
        weight = s ** 2 / s.sum(0)
        s = (weight.t() / weight.sum(1)).t()
        return s

    def loss(self, label_predict, c, sim, label_true):

        # s_refine = self.refined_subspace_affinity(label_predict)
        # loss_class = F.kl_div(label_predict.log(), s_refine.data, reduction='batchmean')

        # make sure final converge to one-hot
        label_predict_norm = F.normalize(label_predict, p=2, dim=0)
        pui = torch.matmul(label_predict_norm.T,label_predict_norm)
        pui.fill_diagonal_(0)
        loss_pui = torch.sum(pui)

        loss_class = compute_negative_entropy_loss(label_predict)
        # distill from C to classfication
        loss_c, loss_info = contrastive(label_predict, c, sim, label_true)

        loss_cluster = empty_cluster_penalty_loss(assignment_matrix = label_predict)

        loss_info.add("loss_class", loss_class)
        loss_info.add("loss_c", loss_c)
        loss_info.add("loss_pui", loss_pui)

        # loss = loss_c + self.eta*(loss_class + loss_cluster) + 0.001*loss_pui
        loss = loss_c + self.eta*(loss_class + loss_cluster)

        return loss, loss_info

    def train_stage_2(self, train_loader, teacher, dataset, epochs):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=2e-6)
        # optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=2e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.01,
                                                         min_lr=1e-10, eps=1e-10, verbose=True)
        # Load the state dictionary from the file
        state_dict = torch.load("checkpoint/teacher_checkpoint.pt")
        teacher.load_state_dict(state_dict)
        teacher.eval()
        self.classifier._initialize_weights()

        for epoch in range(epochs):
            total_loss = 0
            total_loss_class = 0
            total_loss_c = 0
            total_loss_pui = 0
            wrong_positive_rate = 0
            wrong_negative_rate = 0
            positive_sample_number = 0
            negative_sample_number = 0

            for batch_idx, (data, label, x_pca) in enumerate(train_loader):
                similarity_matrix = cos_sim(data)
                z0 = torch.zeros((data.shape[0], data.shape[0]), device=data.device)
                X, x_rescon, h, y_i, c_i, z_s = teacher.forward(data, z0)
                label_predict = self.forward(data)
                loss, loss_info = self.loss(label_predict=label_predict, c=c_i, sim = similarity_matrix, label_true=label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                wrong_positive_rate = wrong_positive_rate + loss_info.wrong_positive_rate
                wrong_negative_rate = wrong_negative_rate + loss_info.wrong_negative_rate
                positive_sample_number = positive_sample_number + loss_info.positive_sample_number
                negative_sample_number = negative_sample_number + loss_info.negative_sample_number
                total_loss += loss
                total_loss_class += loss_info.loss_class
                total_loss_c += loss_info.loss_c
                total_loss_pui += loss_info.loss_pui

            if self.stopping_2.early_stop == False:
                self.stopping_2(total_loss / (batch_idx + 1))
                if self.stopping_2.early_stop == True:
                    torch.save(self.state_dict(), 'checkpoint/model_checkpoint.pt')
                    self.load_state_dict(torch.load('checkpoint/model_checkpoint.pt'))
                    acc, nmi, kappa = self.envaluate(x=dataset.train, y=dataset.y)
                    print("acc:", acc, "nmi:", nmi, "kappa", kappa)
                    break

            logger.log_variables('Training',
                                 [
                                     "loss_class", total_loss_class / (batch_idx + 1),
                                     "sloss", total_loss / (batch_idx + 1),
                                     "loss_c", total_loss_c / (batch_idx + 1),
                                     "loss_c", total_loss_c / (batch_idx + 1),
                                     "swrong_positive_rate", wrong_positive_rate / (batch_idx + 1),
                                     "swrong_negative_rate", wrong_negative_rate / (batch_idx + 1),
                                     "spositive_sample_number", positive_sample_number / (batch_idx + 1),
                                     "snegative_sample_number", negative_sample_number / (batch_idx + 1),
                                     "loss_pui", total_loss_pui / (batch_idx + 1),
                                 ],
                                 epoch, same_window=True)

            #         envaluate every 10 epochs
            if epoch % 10 == 0 or epoch == epochs - 1:
                acc, nmi, kappa = self.envaluate(x=dataset.train, y=dataset.y)
                print("acc:", acc, "nmi:", nmi, "kappa", kappa)

            # scheduler.step(total_loss/(batch_idx+1))
            print("current learning rate:", optimizer.param_groups[0]['lr'])

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

    def envaluate(self, x, y):
        self.eval()
        with torch.no_grad():
            y = y.cpu().numpy()
            label_predict = self.forward(x)
            # label_predict = F.softmax(label_predict, dim=1)
        label_predict = label_predict.detach().cpu().numpy()
        label_predict = np.argmax(label_predict, axis=1)

        # number of class and number of data points
        num_class, num_datapoint = np.unique(label_predict, return_counts=True)
        print("num_class", num_class, "num_datapoint", num_datapoint)

        y_best, acc, kappa, nmi, ari, pur, ca = cluster_accuracy(y_true=y, y_pre=label_predict, return_aligned=True)
        return acc, nmi, kappa
