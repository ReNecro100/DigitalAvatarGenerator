import torch
import torch.nn as nn


class AdaLIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(AdaLIN, self).__init__()
        self.eps = eps

        # Обучаемый параметр gamma (в статье - гамма)
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        nn.init.constant_(self.gamma, 0.5)  # Начинаем с равновесия 50/50

        # Параметры для аффинного преобразования
        self.weight = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        nn.init.constant_(self.weight, 1)
        nn.init.constant_(self.bias, 0)

    def forward(self, x):
        # 1. Instance Norm (по пространству)
        in_mean = x.mean(dim=(2, 3), keepdim=True)
        in_std = x.std(dim=(2, 3), keepdim=True) + self.eps
        out_in = (x - in_mean) / in_std

        # 2. Layer Norm (по каналам и пространству)
        ln_mean = x.mean(dim=(1, 2, 3), keepdim=True)
        ln_std = x.std(dim=(1, 2, 3), keepdim=True) + self.eps
        out_ln = (x - ln_mean) / ln_std

        # 3. Смешиваем с помощью gamma
        out = self.gamma * out_ln + (1 - self.gamma) * out_in

        # 4. Аффинное преобразование
        out = self.weight * out + self.bias

        return out