from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
import os
import shutil

def clear_log_dir(log_dir):
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)


class TensorBoardLogger:
    def __init__(self, log_dir='/home/xianlli/code/0711_runs'):
        # clear_log_dir(log_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_folder_path = os.path.join(log_dir, f"runs_{timestamp}")
        self.writer = SummaryWriter(new_folder_path)

    def log_variables(self, window_name, variables, step, same_window=True):
        """
        Log selected variables to TensorBoard.

        Args:
            window_name (str): base name of the TensorBoard pane.
            variables (list): list structured as [name1, value1, name2, value2, ...].
            step (int): current training step (e.g., epoch).
            same_window (bool): whether to plot all variables under the same window name.
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