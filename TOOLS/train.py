from torch.utils.data import DataLoader
import time

from TOOLS.Student import Student
from TOOLS.Teacher import Teacher
from TOOLS.draw_result_Plt import draw_prediction_with_plt
from TOOLS.get_data import Load_my_Dataset
import torch


def train_clustering(args, device):
    if args.dataset == 'Houston':
        dataset = Load_my_Dataset("/home/xianlli/dataset/HSI/Houston/Houston_corrected.mat",
                                  "/home/xianlli/dataset/HSI/Houston/Houston_gt.mat", patch_size=args.patch_size, band_number=args.n_input,
                                  device=device)
        image_size = [130,130]
    elif args.dataset == 'Trento':
        dataset = Load_my_Dataset("/home/xianlli/dataset/HSI/trento/Trento.mat",
                                  "/home/xianlli/dataset/HSI/trento/Trento_gt.mat", patch_size=args.patch_size, band_number=args.n_input,
                                  device=device)
        image_size = [166,300]


    elif args.dataset == 'Urban':
        dataset = Load_my_Dataset("/home/xianlli/dataset/HSI/urban/Urban_corrected.mat",
                                  "/home/xianlli/dataset/HSI/urban/Urban_gt.mat", patch_size=args.patch_size, band_number=args.n_input,
                                  device=device)
        image_size = [150,160]


    n_cluster = len(torch.unique(dataset.y))
    args.n_cluster = n_cluster
    print(args)
    print("the number of samples: ", len(dataset.y), torch.unique(dataset.y,return_counts=True))
    # train teacher network
    # pre_train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    teacher = Teacher(args=args).to(device=device)

    # teacher.pretrain_ae(train_loader=pre_train_loader, num_epochs=50, path="checkpoint/model_checkpoint_pre.pt", trainable=True)
    start = time.time()
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True)
    teacher.train_stage_1(train_loader=train_loader, dataset=dataset, epochs=200)
    torch.cuda.empty_cache()

    # Train student
    student = Student(args=args).to(device=device)
    acc, nmi, kappa, ca, y_best = student.train_stage_2(train_loader=train_loader, teacher=teacher, dataset=dataset,lr=args.lr, epochs=200)
    print("clustering_result:", "acc:", acc, "nmi:", nmi, "kappa:", kappa, "ca", ca)
    path_true = "/home/xianlli/code/My_TGRS/result/Ours/" + args.dataset + "/" + "label.pdf"
    path_pred= "/home/xianlli/code/My_TGRS/result/Ours/" + args.dataset + "/" + "predict_acc" + str(
            acc) + ".pdf"
    draw_prediction_with_plt(location=dataset.index, pred=y_best, y_true=(dataset.y).cpu().numpy(), image_size=image_size, path_pred=path_pred, path_true=path_true)
    end = time.time()
    print("Elapsed time: {}".format(end - start))
