import matplotlib.pyplot as plt
import numpy as np
import os

color_list_bgr = [
    [255, 0, 0],  # Red
    [0, 255, 0],  # Green
    [0, 0, 255],  # Blue
    [255, 255, 0],  # Yellow
    [128, 0, 128],  # Purple
    [255, 165, 0],  # Orange
    [255, 192, 203],  # Pink
    [64, 224, 208],  # Turquoise
    [165, 42, 42],  # Brown
    [128, 128, 128]  # Gray
]

color_list_rgb = [[b, g, r] for [r, g, b] in color_list_bgr]

# convert color values to the [0, 1] range if needed
color_list = [[c[0] / 255.0, c[1] / 255.0, c[2] / 255.0] for c in color_list_rgb]


def draw_prediction_with_plt(location, pred, y_true, image_size, path_pred, path_true):
    class_label = np.unique(y_true)

    y_true_show = np.zeros([image_size[0], image_size[1], 3])
    for i in range(len(class_label)):
        location_i = np.where(y_true == class_label[i])
        location_y = location[0][location_i]
        location_x = location[1][location_i]
        y_true_show[location_y, location_x, :] = color_list[i]

    y_pred_show = np.zeros([image_size[0], image_size[1], 3])
    for i in range(len(class_label)):
        location_i = np.where(pred == class_label[i])
        location_y = location[0][location_i]
        location_x = location[1][location_i]
        y_pred_show[location_y, location_x, :] = color_list[i]

    # render the ground-truth map
    plt.imshow(y_true_show)
    plt.axis('off')
    os.makedirs(path_true[:-4], exist_ok=True)
    plt.savefig(path_true, bbox_inches='tight', pad_inches=0, format='pdf')

    # render the predicted map
    plt.imshow(y_pred_show)
    plt.axis('off')
    os.makedirs(path_true[:-4], exist_ok=True)
    plt.savefig(path_pred, bbox_inches='tight', pad_inches=0, format='pdf')
