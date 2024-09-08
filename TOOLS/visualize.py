from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
import os
import shutil

def clear_log_dir(log_dir):
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)


class TensorBoardLogger:
    def __init__(self, log_dir='runs'):
        # clear_log_dir(log_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_folder_path = os.path.join(log_dir, f"runs_{timestamp}")
        self.writer = SummaryWriter(new_folder_path)

    def log_variables(self, window_name, variables, step, same_window=True):
        """
        记录指定变量到 TensorBoard

        参数:
        window_name (str): 窗口名称
        variables (list): 变量列表
        step (int): 当前的步数（如 epoch 数）
        same_window (bool): 是否在同一个窗口中显示所有变量
        """
        for i in range(0, len(variables), 2):
            var_name = variables[i]
            var_value = variables[i + 1]
            # var_name = f"Variable_{i}" if len(variables) > 1 else "Variable"
            if same_window:
                self.writer.add_scalar(f"{window_name}/{var_name}", var_value, step)
            else:
                self.writer.add_scalar(f"{window_name}_{var_name}", var_value, step)


    def close(self):
        self.writer.close()