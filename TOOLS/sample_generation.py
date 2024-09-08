import torch
from torch_scatter import scatter_mean
import torch.nn.functional as F
import torch_scatter as scatter

def generate_negative_sample(Cn, Cp):
    Cp = expand_positve_mask(Cp, level=5)
    Cp = (Cp>0).float()

    Cp = torch.matmul(Cp.T, Cp)
    Cn = torch.logical_and(Cn, torch.logical_not(Cp))


    return Cn

# def compute_mean_from_indices(a, b):
#     """
#     Compute the mean of selected values from tensor `a` based on indices from tensor `b`.
#
#     Parameters:
#     a (torch.Tensor): A 2D tensor with shape [N, M].
#     b (torch.Tensor): A 2D tensor with shape [P, Q] containing indices.
#
#     Returns:
#     torch.Tensor: A tensor containing the means with shape [N, P].
#     """
#     # Prepare the row indices and column indices for indexing
#     rows = torch.arange(a.size(0)).unsqueeze(1)  # Shape: [N, 1]
#     cols = b  # Shape: [P, Q]
#
#     # Repeat tensor `a` to match the indexing shape
#     expanded_a = a.unsqueeze(1).expand(-1, b.size(0), -1)  # Shape: [N, P, M]
#
#     # Indexing tensor to match the required shape
#     indices = cols.unsqueeze(0)  # Shape: [1, P, Q]
#     indices = indices.expand(a.size(0), -1, -1)  # Shape: [N, P, Q]
#
#     # Gather values using the indices
#     selected_values = torch.gather(expanded_a, 2, indices)  # Shape: [N, P, Q]
#
#     # Compute the mean of the selected values along the last dimension
#     c = selected_values.float().mean(dim=2)  # Shape: [N, P]
#
#     return c

# def generate_positive_sample(Cp, sim):
#     Cp_n = F.normalize(Cp, dim=0)
#     similarity_matrix = torch.matmul(Cp_n.T, Cp_n)
#     similarity_s, top_5 = similarity_matrix.topk(5, dim=1)
#     # similarity_s, top_5 = sim.topk(5, dim=1)
#
#     # 获取 Cp 的形状
#     # similarity_s_show = similarity_s.detach().cpu().numpy()
#
#     # Perform the indexing operation with broadcasting
#     extracted_matrix = compute_mean_from_indices(Cp, top_5)
#
#     extracted_matrix_show = extracted_matrix.cpu().numpy()
#
#     extracted_matrix = ((extracted_matrix>0.4)).float()
#
#     # extracted_matrix = torch.logical_and(extracted_matrix,Cp)
#
#     extracted_matrix = (torch.matmul(extracted_matrix.T,extracted_matrix)>0)
#
#     Cp = torch.logical_or(extracted_matrix,Cp).int()
#
#     # extracted_matrix_show = extracted_matrix.cpu().numpy()
#
#     return Cp


def generate_positive_samples(Cp):
    Cp_EP = expand_positve_mask_(Cp, level=5)
    Cp_EP = torch.logical_or(Cp,Cp_EP).int()
    Cp_show = Cp.cpu().numpy()
    return Cp_EP

def expand_positve_mask_(positive_mask, level):
    expand_mask = positive_mask
    for i in range(level):
        positive_mask = torch.matmul(positive_mask.float(), positive_mask)
        positive_mask = (positive_mask>1).float()
        # expand_mask = torch.where(positive_mask > 0, torch.tensor(1), positive_mask)
    return positive_mask


def expand_positve_mask(positive_mask, level):
    expand_mask = positive_mask
    for i in range(level):
        positive_mask = torch.matmul(positive_mask.float(), expand_mask)
        expand_mask = (positive_mask>0).float()
        # expand_mask = torch.where(positive_mask > 0, torch.tensor(1), positive_mask)
    return positive_mask


def robust_sample(Cp):
    mask_t, _ = torch.sort(Cp, dim=0, descending=True)
    C_sum = torch.sum(Cp, dim=0)
    positive_thred = C_sum * 0.5
    mask_sum = torch.cumsum(mask_t, dim=0)
    mask_select = torch.where(mask_sum <= positive_thred.unsqueeze(0), torch.ones_like(Cp), torch.zeros_like(Cp))

    # Ensure the input matrix is on the correct device and has the correct dtype
    device = Cp.device
    # dtype = Cp.dtype

    # Sort each column in descending order
    # mask, _ = torch.sort(Cp, dim=0, descending=True)

    # Calculate the robust representation by multiplying the matrix with itself
    Cp = torch.matmul(Cp, Cp)
    # Cp_show = Cp.cpu().numpy()

    # Use argsort to get the row indices sorted by columns in descending order
    Cp_sort = torch.argsort(Cp, dim=0, descending=True)
    # Cp_sort_show = Cp_sort.cpu().numpy()

    # Get the shape of the matrix
    num_rows, num_cols = Cp.shape

    # Create a tensor to hold the sorted indices
    sorted_indices = torch.zeros((num_rows, num_cols, 2), dtype=torch.long, device=device)

    # Set column indices
    col_indices = torch.arange(num_cols, device=device).repeat(num_rows, 1)

    # Set row indices and column indices into sorted_indices
    sorted_indices[:, :, 0] = Cp_sort  # Row indices
    sorted_indices[:, :, 1] = col_indices  # Column indices

    # Expand mask to the same shape as the sorted_indices and convert to boolean
    expanded_MASK = mask_select.unsqueeze(-1).expand(-1, -1, 2).bool()

    # Use boolean indexing to filter elements in the matrix
    selected_index = sorted_indices[expanded_MASK].view(-1, 2)

    # Initialize the result matrix with zeros
    result = torch.zeros_like(Cp, device=device)

    # Set the selected indices to 1 in the result matrix
    result[selected_index[:, 0], selected_index[:, 1]] = 1

    return result
