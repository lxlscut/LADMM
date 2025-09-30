import torch
import torch.nn.functional as F
import torch_scatter as scatter

def generate_negative_sample(Cn, Cp):
    Cp = expand_positve_mask(Cp, level=5)
    Cp = (Cp>0).float()
    Cp = torch.matmul(Cp.T, Cp)
    Cn = torch.logical_and(Cn, torch.logical_not(Cp))
    return Cn

def generate_positive_samples(Cp):
    Cp_EP = expand_positve_mask_(Cp, level=5)
    Cp_EP = torch.logical_or(Cp,Cp_EP).int()
    # Cp_show = Cp.cpu().numpy()
    return Cp_EP

def expand_positve_mask_(positive_mask, level):
    expand_mask = positive_mask
    for i in range(level):
        positive_mask = torch.matmul(positive_mask.float(), positive_mask)
        positive_mask = (positive_mask>1).float()
        # expand_mask = torch.where(positive_mask > 0, torch.tensor(1), positive_mask)
    return positive_mask


def expand_positve_mask(positive_mask, level):
    # the number of positive samples
    sum_p = torch.sum(positive_mask)
    # the negative samples should lie in 3times to 10times of positive samples
    sum_n_up = sum_p*10
    sum_n_down = sum_p*3
    expand_mask = positive_mask
    for i in range(level):
        positive_mask = torch.matmul(positive_mask.float(), expand_mask)
        # current_negative_mask = (positive_mask==0).float()
        # if torch.sum(current_negative_mask)<sum_n_down:
        #     return expand_mask
        expand_mask = (positive_mask > 0).float()
    return positive_mask


def set_top_n_to_one(matrix, N):
    """
    Set the locations of the top-N values in the matrix to 1 and keep the rest unchanged.

    Args:
    - matrix (torch.Tensor): input matrix
    - N (int): number of elements that will be set to 1

    Returns:
    - torch.Tensor: resulting mask tensor
    """
    # obtain the indices of the top-N values
    # convert N to a Python int if it is a tensor
    # if isinstance(N, torch.Tensor):
    #     N = N.item()
    # N = int(N)
    top_values, top_indices = torch.topk(matrix.view(-1), N)

    # create a zero matrix with the same shape as the input
    result = torch.zeros_like(matrix)

    # set the selected positions to 1
    result.view(-1)[top_indices] = 1

    return result

def set_bottom_n_to_one(matrix, N):
    """
    Set the locations of the bottom-N values in the matrix to 1 and keep the rest unchanged.

    Args:
    - matrix (torch.Tensor): input matrix
    - N (int): number of elements with the smallest values to flag as 1

    Returns:
    - torch.Tensor: resulting mask tensor
    """
    # obtain the indices of the bottom-N values

    # convert N to a Python int if it is a tensor
    # if isinstance(N, torch.Tensor):
    #     N = N.item()
    # N = int(N)
    # N  = min(N,200000)

    bottom_values, bottom_indices = torch.topk(matrix.view(-1), N, largest=False)

    # create a zero matrix with the same shape as the input
    result = torch.zeros_like(matrix)

    # set the selected positions to 1
    result.view(-1)[bottom_indices] = 1

    return result


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




