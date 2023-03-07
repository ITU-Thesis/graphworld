import numpy as np
import torch
import torch.nn.functional as F

# Copied from https://github.com/Namkyeong/BGRL_Pytorch
class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new
    

def init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

def pad_views(view1data, view2data):
        diff = abs(view2data.x.shape[1] - view1data.x.shape[1])
        if diff > 0:
            smaller_data = view1data if view1data.x.shape[1] < view2data.x.shape[1] else view2data
            smaller_data.x = F.pad(smaller_data.x, pad=(0, diff))
            view1data.x = F.normalize(view1data.x)
            view2data.x = F.normalize(view2data.x)