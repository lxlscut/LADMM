from TOOLS.AE import Ae
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from TOOLS.Contrasitive_loss import Contrasitive_loss
from TOOLS.early_stopping import EarlyStoppingCurveWithSmooth
from TOOLS.evaluation import cluster_accuracy
from TOOLS.loss_function import compute_negative_entropy_loss, empty_cluster_penalty_loss
from TOOLS.tools import cos_sim
from TOOLS.visualize import TensorBoardLogger

logger = TensorBoardLogger()


class classifier(nn.Module):
    def __init__(self, in_dim, out_dim, patch_size):
        super(classifier, self).__init__()
        # classification branch
        self.encoder = nn.Sequential()
        self.encoder.add_module("cov01", nn.Conv2d(in_channels=in_dim, out_channels=128, kernel_size=[1, 1], stride=1,
                                                   padding='same'))
        self.encoder.add_module("bn01", nn.BatchNorm2d(128))
        self.encoder.add_module("relu01", nn.ReLU(True))


        self.encoder.add_module("cov02", nn.Conv2d(in_channels=128, out_channels=64, kernel_size=[3, 3], stride=1,
                                                   padding='same'))
        self.encoder.add_module("bn02", nn.BatchNorm2d(64))
        self.encoder.add_module("relu02", nn.ReLU(True))


        self.encoder.add_module("cov03", nn.Conv2d(in_channels=64, out_channels=32, kernel_size=[3, 3], stride=1,
                                                   padding='same'))
        self.encoder.add_module("bn03", nn.BatchNorm2d(32))
        self.encoder.add_module("relu03", nn.ReLU(True))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=32*patch_size*patch_size, out_features=1024),
            nn.BatchNorm1d(1024),  # add batch normalization
            nn.ReLU(True),
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),  # add batch normalization
            nn.ReLU(True),
            nn.Linear(in_features=512, out_features=out_dim),
            nn.Softmax(dim=-1)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # check whether this is the last fully connected layer
                if m is self.classifier[-3]:  # the third item from the end is the final nn.Linear layer
                    nn.init.xavier_normal_(m.weight)  # use Xavier initialization for the final layer
                else:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # use Kaiming initialization for intermediate layers
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        label_predict = self.classifier(h)
        return label_predict


class Student(nn.Module):
    def __init__(self, args):
        super(Student, self).__init__()
        self.contras = Contrasitive_loss(n_cluster=args.n_cluster)
        self.classifier = classifier(in_dim=args.n_input, out_dim=args.n_cluster,patch_size=args.patch_size)
        self.stopping_2 = EarlyStoppingCurveWithSmooth(min_delta=1e-5, patience=30)
        self.eta = args.eta
    def forward(self, input):
        output = self.classifier(input)
        return output

    def loss(self, label_predict, c, sim, label_true):
        # enforce one-hot prediction
        label_predict_norm = F.normalize(input=label_predict, p=2, dim=0)
        pui = torch.matmul(label_predict_norm.T, label_predict_norm)
        pui.fill_diagonal_(0)
        loss_pui = torch.sum(pui)

        loss_class = compute_negative_entropy_loss(label_predict)
        # distill from C to classfication

        # c_show = c.detach().cpu().numpy()
        loss_c, loss_info = self.contras.contrastive(label_predict, c, sim, label_true)

        loss_cluster = empty_cluster_penalty_loss(assignment_matrix = label_predict)

        loss_info.add("loss_class", loss_class)
        loss_info.add("loss_c", loss_c)
        loss_info.add("loss_pui", loss_pui)
        loss_info.add("loss_cluster", loss_cluster)

        loss = loss_c
        return loss, loss_info

    def train_stage_2(self, train_loader, teacher, dataset, lr, epochs):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=0.0001,
                                                         min_lr=1e-10, eps=1e-10, verbose=True)
        # Load the state dictionary from the file
        state_dict = torch.load("checkpoint/teacher_checkpoint.pt")
        teacher.load_state_dict(state_dict)
        teacher.eval()
        # self.ae.initialize_weights()
        self.classifier._initialize_weights()

        for epoch in range(epochs):
            total_loss = 0
            total_loss_class = 0
            total_loss_pui = 0
            total_loss_c = 0
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
                                     "loss_pui", total_loss_pui / (batch_idx + 1),
                                     "swrong_positive_rate", wrong_positive_rate / (batch_idx + 1),
                                     "swrong_negative_rate", wrong_negative_rate / (batch_idx + 1),
                                     "spositive_sample_number", positive_sample_number / (batch_idx + 1),
                                     "snegative_sample_number", negative_sample_number / (batch_idx + 1),
                                 ],
                                 epoch, same_window=True)
            #         envaluate every 10 epochs
            if epoch % 10 == 0 or epoch == epochs - 1:
                acc, nmi, kappa, ca, y_best = self.envaluate(x=dataset.train, y=dataset.y)
                print("acc:", acc, "nmi:", nmi, "kappa", kappa)

            # scheduler.step(total_loss/(batch_idx+1))
            print("current learning rate:", optimizer.param_groups[0]['lr'])
        return acc, nmi, kappa, ca, y_best

    def predict_batches(self, data_tensor: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
        """
        Perform batched inference on a tensor input.

        Args:
            data_tensor: input tensor shaped as (N, ...), where N is the number of samples.
            batch_size: size of the batch used during inference.
        """
        predictions = []  # collect predictions for each batch

        # compute how many batches are needed
        num_batches = (data_tensor.size(0) + batch_size - 1) // batch_size  # ceil division to cover the last partial batch

        for i in range(num_batches):
            # determine the start and end indices of the current batch
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, data_tensor.size(0))  # clamp to stay within bounds

            # slice the current batch of data
            batch_data = data_tensor[start_idx:end_idx]

            # run inference for the current batch
            x_rescon, h = self.ae(batch_data)
            label_predict = self.classifier(h)
            predictions.append(label_predict)

        # concatenate all batch predictions into a single tensor
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
        return acc, nmi, kappa, ca, y_best
