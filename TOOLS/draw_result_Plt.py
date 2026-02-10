import matplotlib.pyplot as plt
import numpy as np
import os

# -------- 颜色表（保持与你原始代码一致）--------
color_list_bgr = [
    [255, 0, 0],      # Red
    [0, 255, 0],      # Green
    [0, 0, 255],      # Blue
    [255, 255, 0],    # Yellow
    [128, 0, 128],    # Purple
    [255, 165, 0],    # Orange
    [255, 192, 203],  # Pink
    [64, 224, 208],   # Turquoise
    [165, 42, 42],    # Brown
    [128, 128, 128]   # Gray
]
color_list_rgb = [[b, g, r] for [r, g, b] in color_list_bgr]
color_list = [[c[0] / 255.0, c[1] / 255.0, c[2] / 255.0] for c in color_list_rgb]

# 全局关闭图像插值与重采样，确保硬边缘
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.resample'] = False

def _save_rgb_mask_as_figure(mask_rgb, out_path, dpi=600):
    """
    将 HxWx3 的RGB数组以“像素=画布像素”的1:1方式保存，避免任何缩放导致的柔化。
    """
    H, W, _ = mask_rgb.shape
    # 让画布尺寸与像素严格匹配：英寸 = 像素 / DPI
    fig_w, fig_h = W / dpi, H / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    # 去除边距与坐标轴
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.set_axis_off()
    # 确保硬边缘显示
    ax.imshow(mask_rgb, interpolation='nearest')
    # 保存（根据文件后缀自动选择格式；PDF/PNG皆可）
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def draw_prediction_with_plt(location, pred, y_true, image_size, path_pred, path_true, dpi=600):
    """
    无缝替换版：导出硬边缘高分辨率图像（PDF/PNG均可）。
    参数签名与行为保持一致；新增可选参数 dpi（默认600）。
    """
    class_label = np.unique(y_true)

    # 预分配显示图（float，范围0~1）
    H, W = int(image_size[0]), int(image_size[1])
    y_true_show = np.zeros((H, W, 3), dtype=float)
    y_pred_show = np.zeros((H, W, 3), dtype=float)

    # 填色（按 class_label 的顺序使用 color_list）
    for i, cls in enumerate(class_label):
        loc_i_true = np.where(y_true == cls)
        y_idx_true = location[0][loc_i_true]
        x_idx_true = location[1][loc_i_true]
        y_true_show[y_idx_true, x_idx_true, :] = color_list[i]

        loc_i_pred = np.where(pred == cls)
        y_idx_pred = location[0][loc_i_pred]
        x_idx_pred = location[1][loc_i_pred]
        y_pred_show[y_idx_pred, x_idx_pred, :] = color_list[i]

    # 确保输出目录存在
    if os.path.dirname(path_true):
        os.makedirs(os.path.dirname(path_true), exist_ok=True)
    if os.path.dirname(path_pred):
        os.makedirs(os.path.dirname(path_pred), exist_ok=True)

    # 分别保存，避免叠画与状态污染
    _save_rgb_mask_as_figure(y_true_show, path_true, dpi=dpi)
    _save_rgb_mask_as_figure(y_pred_show, path_pred, dpi=dpi)
