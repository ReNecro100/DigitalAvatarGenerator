import torch
import torch.nn as nn

from train.models.CAM import CAMMap


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN дискриминатор (строго по таблице).
    Использует BCELoss, поэтому на выходе стоит Sigmoid.
    """

    def __init__(self, input_nc=3, ndf=64):
        super(PatchGANDiscriminator, self).__init__()

        # Conv1: 64 фильтра, ядро 4x4, шаг 2, паддинг 1. Без нормализации.
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Conv2: 128 фильтров. Заменили BatchNorm на InstanceNorm.
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Conv3: 256 фильтров.
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Conv4: 512 фильтров, но stride=1 (размер не уменьшается).
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.cam_map = CAMMap(ndf * 8)

        # Conv5: 1 фильтр (итоговая карта), stride=1.
        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        cam_map = self.cam_map(x)
        x = self.conv5(x)
        return (x, cam_map)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# # Блок для проверки
# if __name__ == "__main__":
#     # Подаем картинку 256x256
#     dummy_input = torch.randn(1, 3, 256, 256)
#     discriminator = PatchGANDiscriminator()
#     output = discriminator(dummy_input)
#
#     print(discriminator.count_parameters())
#     print(f"Вход: {dummy_input.shape}")
#     print(f"Выход: {output.shape}")
#     print(f"Мин. значение на выходе: {output.min().item():.4f}")
#     print(f"Макс. значение на выходе: {output.max().item():.4f}")