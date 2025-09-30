import numpy as np
import torch
from sklearn.preprocessing import scale, normalize, minmax_scale, robust_scale
from osgeo import gdal


def get_data(img_path, label_path):
    if img_path[-3:] == 'tif':
        img_data = gdal.Open(img_path).ReadAsArray()
        label_data = gdal.Open(label_path).ReadAsArray()
        img_data = np.transpose(img_data, [1, 2, 0])
        return img_data, label_data
    elif img_path[-3:] == 'mat':
        import scipy.io as sio
        img_mat = sio.loadmat(img_path)
        img_keys = img_mat.keys()
        img_key = [k for k in img_keys if k != '__version__' and k != '__header__' and k != '__globals__']

        if label_path is not None:
            gt_mat = sio.loadmat(label_path)
            gt_keys = gt_mat.keys()
            gt_key = [k for k in gt_keys if k != '__version__' and k != '__header__' and k != '__globals__']
            return img_mat.get(img_key[0]).astype('float32'), gt_mat.get(gt_key[0]).astype('int8')
        return img_mat.get(img_key[0]).astype('float32'), img_mat.get(img_key[1]).astype('int8')


"""Default stride is 1 when it is not explicitly specified."""


# @njit()
def get_data_patch(data, patch_size):
    patch_w = patch_size[0]
    patch_h = patch_size[1]

    # compute padding sizes
    pad_h = int((patch_h - 1) / 2)
    pad_w = int((patch_w - 1) / 2)
    # pad_h = pad_h.astype(np.int16)
    # pad_w = pad_w.astype(np.int16)

    res = np.zeros((data.shape[0], data.shape[1], patch_w, patch_h, data.shape[2]))

    # obtain the padded image
    data_ = np.pad(data, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'edge')

    for i in range(pad_h, data.shape[0] - pad_h):
        for j in range(pad_w, data_.shape[1] - pad_w):
            patch = data_[i - pad_h:i + pad_h + 1, j - pad_w:j + pad_w + 1, :]
            res[i - pad_h, j - pad_w, :, :, :] = patch
    return res


def get_patch(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):
    """
    Extract sliding windows from an array using numpy stride tricks.

    Args:
        array: array to be windowed.
        window: window size; a scalar targets the last dimension, a tuple defines the block shape.
        asteps: strides between adjacent blocks.
        wsteps: strides within each block.
        axes: axes along which the window is applied.
        toend: whether to move block axes to the end of the output shape.

    Returns:
        numpy.ndarray: view of the input containing the extracted patches.
    """
    array = np.asarray(array)  # redundant?
    orig_shape = np.asarray(array.shape)

    # convert scalars to arrays while leaving arrays unchanged
    window = np.atleast_1d(window).astype(int)

    # derive the block dimensions and their sizes
    if axes is not None:
        axes = np.atleast_1d(axes)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axes] = size
        window = w

    # window must be a one-dimensional array
    if window.ndim > 1:
        raise ValueError("window must be one-dimenional.")
    if np.any(window < 0):
        raise ValueError("Each dimension length must be greater than one.")
    # This function extracts sub-blocks, so each block must be no larger than the original array
    if len(array.shape) < len(window):
        raise ValueError("Window dimensionality must be less than or equal to the input array.")

    _asteps = np.ones_like(orig_shape)
    # asteps represent the stride between adjacent blocks
    if asteps is not None:
        asteps = np.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError('asteps must be a one-dimensional list of strides for each window dimension')
        if len(asteps) > array.ndim:
            raise ValueError('block dims bigger than array dims')
        # debugging helper
        _asteps[-len(asteps):] = asteps
        if np.any(asteps < 1):
            raise ValueError("Strides must be greater than or equal to 1.")
    asteps = _asteps

    _wsteps = np.ones_like(window)
    if wsteps is not None:
        wsteps = np.atleast_1d(wsteps)
        # wsteps control sampling strides inside the window
        if wsteps.shape != window.shape:
            raise ValueError("Invalid dimension specification")
        if np.any(wsteps <= 0):
            raise ValueError("All strides must be greater than 0")
        _wsteps[:] = wsteps
        # steps should be at least 1
        _wsteps[window == 0] = 1
    wsteps = _wsteps

    # window size multiplied by stride must not exceed the array dimensions
    if np.any(orig_shape[-len(window):] < window * wsteps):
        raise ValueError('window*wsteps larger than array in at least one demension')

    new_shape = orig_shape
    _window = window.copy()
    _window[_window == 0] = 1

    new_shape[-len(window):] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    new_shape[new_shape < 1] = 1
    shape = new_shape

    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window):] * wsteps

    if toend:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window)] = window
        _window = _.copy()
        _[-len(window)] = new_strides
        _new_strides = _
        new_shape = np.zeros(len(shape) * 2, dtype=int)
        new_strides = np.zeros(len(shape) * 2, dtype=int)

        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides

    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]

    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)


def get_HSI_patches(x, gt, ksize, stride=(1, 1), padding='reflect', is_index=False, is_labeled=True):
    """
    :param x: inputs data
    :param gt: label data
    :param ksize: kernal_size
    :param stride: stride value
    :param padding: the mode of padding
    :param is_index:
    :param is_labeled:
    :return:
    """
    # np.ceil performs a ceiling operation
    new_height = np.ceil(x.shape[0] / stride[0])
    new_width = np.ceil(x.shape[1] / stride[1])
    band = x.shape[2]

    pad_needed_height = (new_height - 1) * stride[0] + ksize[0] - x.shape[0]
    pad_needed_width = (new_width - 1) * stride[1] + ksize[1] - x.shape[1]

    pad_top = int(pad_needed_height / 2)
    pad_down = int(pad_needed_height - pad_top)
    pad_left = int(pad_needed_width / 2)
    pad_right = int(pad_needed_width - pad_left)

    # pad every dimension of the 3D image
    x = np.pad(x, ((pad_top, pad_down), (pad_left, pad_right), (0, 0)), padding)
    gt = np.pad(gt, ((pad_top, pad_down), (pad_left, pad_right)), padding)

    n_row, n_clm, n_band = x.shape

    x = np.reshape(x, (n_row, n_clm, n_band))
    y = np.reshape(gt, (n_row, n_clm))
    # treat the window as a tuple
    ksize = (ksize[0], ksize[1])

    # extract patches
    x_patches = get_patch(x, ksize, axes=(1, 0))
    y_patches = get_patch(y, ksize, axes=(1, 0))

    # use the center element of the 7x7 label patch as the ground-truth label
    i_1, i_2 = int((ksize[0] - 1) // 2), int((ksize[1] - 1) // 2)

    min_val = np.min(y)

    nonzero_index = np.where(y_patches[:, :, i_1, i_2] > min_val)
    all_index = np.where(y_patches[:, :, i_1, i_2] > min_val - 1)

    if is_labeled is False:
        x_patches = x_patches.reshape(
            [x_patches.shape[0] * x_patches.shape[1], x_patches.shape[2], x_patches.shape[3], x_patches.shape[4]])
        x_patches = np.transpose(x_patches, [0, 2, 3, 1])
        y_patches = y_patches[:, :, i_1, i_2]
        return x_patches, y_patches, nonzero_index, all_index

    x_patches_nonzero = x_patches[nonzero_index]
    x_patches_nonzero = np.transpose(x_patches_nonzero, [0, 2, 3, 1])
    y_patches_nonzero = (y_patches[:, :, i_1, i_2])[nonzero_index]

    if is_index is True:
        return x_patches_nonzero, y_patches_nonzero, nonzero_index

    y_patches_nonzero = standardize_label(y_patches_nonzero)

    print('x_patches shape: %s, labels: %s' % (x_patches.shape, np.unique(y)))

    # x_patches_nonzero = np.transpose(x_patches_nonzero, [0, 3, 1, 2])
    y_patches_nonzero = y_patches_nonzero.flatten()
    return x_patches_nonzero, y_patches_nonzero, nonzero_index


"""The original y values are non-contiguous integers."""


def standardize_label(y):
    """
    standardize the classes label into 0-k
    :param y:
    :return:
    """
    import copy
    classes = np.unique(y)
    standardize_y = copy.deepcopy(y)
    for i in range(classes.shape[0]):
        standardize_y[np.nonzero(y == classes[i])] = i
    return standardize_y


class Load_my_Dataset():

    def __init__(self, image_path, label_path, patch_size, band_number, device):
        X, Y = get_data(image_path, label_path)

        # only for trento dataset
        # X, Y = X[ :,100 :400, :], Y[:,100:400,]

        n_row, n_column, n_band = X.shape

        if not band_number == n_band:
            # perform PCA
            from sklearn.decomposition import PCA
            n_components = band_number
            pca = PCA(n_components)
            X = scale(X.reshape(n_row * n_column, n_band))
            X = pca.fit_transform(X.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_components))
        x_train, y_patches, index = get_HSI_patches(x=X, gt=Y, ksize=(patch_size, patch_size), is_labeled=True)
        x_train = np.transpose(x_train, axes=(0, 3, 1, 2))
        n_samples, n_channel, n_row, n_col  = x_train.shape
        x_train = scale(x_train.reshape((n_samples, -1))).reshape((n_samples,n_channel, n_row, n_col))


        # perform PCA
        from sklearn.decomposition import PCA
        n_components = 300
        pca = PCA(n_components)
        X_pca = pca.fit_transform(x_train.reshape((n_samples, -1)))
        # self.y, self.train, self.index = y_patches, x_train, index

        self.y, self.train, self.X_pca, self.index = torch.from_numpy(np.array(y_patches)).to(device), torch.from_numpy(
            np.array(x_train)).to(device), torch.from_numpy(
            np.array(X_pca)).to(device), index

    def __len__(self):
        return self.train.shape[0]

    def __getitem__(self, idx):
        return self.train[idx], self.y[idx], self.X_pca[idx]
