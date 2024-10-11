import torch.nn.functional as F
from TOOLS.info import VariableContainer
from TOOLS.loss_function import thrC
from TOOLS.sample_generation import generate_negative_sample, robust_sample,generate_positive_samples
import torch



def contrastive(label_predict, C, sim, label_true):
    """
    :param label_predict: the output of classifier network
    :param C: the self-representation matrix
    :return:
    """

    loss_info = VariableContainer()
    S = (torch.abs(C) + torch.abs(C.T)) / 2
    S = S.detach()
    S_show = S.cpu().numpy()

    Cp, Cn = thrC(S, ro_p=0.5, ro_n=0.10)
    C_p = Cp.detach().cpu().numpy()

    # if sim is not None:
    #     CosP = sim>0.95
    #     Cp = torch.logical_or(Cp, CosP).to(torch.float32)
    #     Cn = generate_negative_sample(Cn, Cp)
    #     CosN = sim<0.5
    #     Cn = torch.logical_and(Cn, CosN)




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

    # Cp.fill_diagonal_(False)
    # Cn.fill_diagonal_(False)
    # label_predict_norm = F.normalize(label_predict, p=2, dim=1)
    # similarity_predict = torch.matmul(label_predict_norm, label_predict_norm.T)
    #
    # # positive_sum = torch.sum(similarity_predict * Cp, dim=0)
    # # negative_sum = torch.sum(similarity_predict * Cn, dim=0)
    #
    # positive_sum = torch.sum(similarity_predict * Cp.T, dim=1)
    # negative_sum = torch.sum(similarity_predict * Cn.T, dim=1)
    #
    # temp = (positive_sum + 1e-6) / (positive_sum + negative_sum+1)
    # loss_ = -torch.log(temp)
    # loss = torch.mean(loss_)

    Cp.fill_diagonal_(False)
    Cn.fill_diagonal_(False)
    MASK = torch.logical_or(Cp, Cn)

    label_predict_norm = F.normalize(label_predict, p=2, dim=1)
    similarity_predict = torch.matmul(label_predict_norm, label_predict_norm.T) / 0.5

    max_similarity, _ = torch.max(similarity_predict, dim=1, keepdim=True)
    s_stable = similarity_predict - max_similarity

    sim_exp = torch.exp(s_stable)

    sim_exp_m = sim_exp * MASK.T

    sim_exp_con = s_stable - torch.log(torch.sum(sim_exp_m, dim=1, keepdim=True)+1e-6)

    pos_sum = torch.sum(sim_exp_con * Cp.T, dim=1)

    sample_numer = torch.sum(Cp.T, dim=1)
    sample_numer[sample_numer == 0] = 1
    if (sample_numer == 0).any():
        # 执行一些处理，比如避免除以 0 的操作
        raise ValueError('Division by zero detected in mask_pos_pairs')

    mean_con = pos_sum / (sample_numer+1)

    loss = -torch.mean(mean_con)

    return loss, loss_info
