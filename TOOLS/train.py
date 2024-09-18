from torch.utils.data import DataLoader
import time

from TOOLS.Student import Student
from TOOLS.Teacher import Teacher
from TOOLS.get_data import Load_my_Dataset
import torch


def train_clustering(args, device):
    if args.dataset == 'Houston':
        dataset = Load_my_Dataset("/home/xianlli/dataset/HSI/Houston/Houston_corrected.mat",
                                  "/home/xianlli/dataset/HSI/Houston/Houston_gt.mat", patch_size=args.patch_size, band_number=args.n_input,
                                  device=device)
    elif args.dataset == 'Trento':
        dataset = Load_my_Dataset("/home/xianlli/dataset/HSI/trento/Trento.mat",
                                  "/home/xianlli/dataset/HSI/trento/Trento_gt.mat", patch_size=args.patch_size, band_number=args.n_input,
                                  device=device)

    elif args.dataset == 'Pavia':
        dataset = Load_my_Dataset("/home/xianlli/dataset/HSI/pavia/PaviaU.mat",
                                  "/home/xianlli/dataset/HSI/pavia/PaviaU_gt.mat", patch_size=args.patch_size, band_number=args.n_input,
                                  device=device)

    n_cluster = len(torch.unique(dataset.y))
    args.n_cluster = n_cluster
    print(args)

    # train teacher network
    pre_train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    teacher = Teacher(args=args).to(device=device)

    # teacher.pretrain_ae(train_loader=pre_train_loader, num_epochs=50, path="checkpoint/model_checkpoint_pre.pt", trainable=True)
    start = time.time()
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True)
    teacher.train_stage_1(train_loader=train_loader, dataset=dataset, epochs=200)
    torch.cuda.empty_cache()

    # Train student
    student = Student(args=args).to(device=device)
    acc, nmi, kappa = student.train_stage_2(train_loader=train_loader, teacher=teacher, dataset=dataset,lr=args.lr, epochs=500)
    print("clustering_result:", "acc:", acc, "nmi:", nmi, "kappa:", kappa)
    end = time.time()
    print("Elapsed time: {}".format(end - start))
