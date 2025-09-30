import visdom
import numpy as np


class FunctionMonitor:
    """Class to monitor function calls and update Visdom plot accordingly."""
    def __init__(self, env_name='main', var_name='Value', title_name='Function Call Monitoring'):
        self.vis = visdom.Visdom()
        self.env = env_name
        self.counter = 0
        self.var_name = var_name
        self.title_name = title_name
        self.plot = None
        assert self.vis.check_connection(), "Failed to connect to Visdom server."

    def update_plot(self, y):
        """Update plot with new data point."""
        if self.plot is None:
            self.plot = self.vis.line(X=np.array([self.counter]), Y=np.array([y]),
                                      env=self.env, opts=dict(
                title=self.title_name,
                xlabel='Calls',
                ylabel=self.var_name
            ))
        else:
            self.vis.line(X=np.array([self.counter]), Y=np.array([y]), env=self.env,
                          win=self.plot, update='append')
        self.counter += 1

# Usage example
monitor = FunctionMonitor(var_name='Random Value', title_name='Random Value Monitoring')
