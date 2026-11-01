import torch
import torch.nn as nn

class CAMMap(nn.Module):
    def __init__(self, dim):
        super(CAMMap, self).__init__()
        self.cam_loss = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.cam_loss(x).view(x.size(0), 1, 1, 1) * x