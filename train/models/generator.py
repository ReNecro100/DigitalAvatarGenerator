import torch
import torch.nn as nn
from train.models.AdaLIN import *
from train.models.CAM import CAMMap


class ResNetBlock(nn.Module):
    """Один Residual блок (сохраняет черты лица благодаря skip-connection)"""

    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        # Прибавляем исходный x к результату (skip connection)
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """
    Генератор (Фото -> Мультик или Мультик -> Фото).
    Явно прописанные слои, без циклов.
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(ResNetGenerator, self).__init__()

        # 1. Начальный слой (ngf = 64 фильтра)
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        )

        # 2. Даунсэмплинг (уменьшаем разрешение в 2 раза, потом еще в 2)
        self.down1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # 3. Девять Residual блоков (работаем с разрешением 64x64 и 256 фильтрами)
        self.res1 = ResNetBlock(ngf * 4)
        self.res2 = ResNetBlock(ngf * 4)
        self.res3 = ResNetBlock(ngf * 4)
        self.res4 = ResNetBlock(ngf * 4)
        self.res5 = ResNetBlock(ngf * 4)
        self.res6 = ResNetBlock(ngf * 4)
        self.res7 = ResNetBlock(ngf * 4)
        self.res8 = ResNetBlock(ngf * 4)
        self.res9 = ResNetBlock(ngf * 4)

        # 2.5 CAM-карта
        self.cam_map = CAMMap(ngf * 4)

        # 4. Апсэмплинг (увеличиваем разрешение обратно)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # Сначала плавно растягиваем
            nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=1, bias=False),  # Потом сворачиваем
            AdaLIN(ngf * 2),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # Сначала плавно растягиваем
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1, bias=False),  # Потом сворачиваем
            AdaLIN(ngf),
            nn.ReLU(True)
        )

        # 5. Финальный слой (выход 3 канала RGB, значения от -1 до 1)
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        # 1. Начало
        x = self.initial(x)

        # 2. Сжатие
        x = self.down1(x)
        x = self.down2(x)

        # 3. Магия стилизации (9 блоков)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.res9(x)

        # 3.5 CAM-карта
        cam = self.cam_map(x)

        # 4. Разжатие
        x = self.up1(x)
        x = self.up2(x)

        # 5. Выход
        return (self.final(x), cam)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

#Блок для проверки
# if __name__ == "__main__":
#     # Создаем случайную картинку 256x256 (1 картинка, 3 канала)
#     dummy_input = torch.randn(1, 3, 256, 256)
#
#     # Инициализируем генератор
#     generator = ResNetGenerator()
#
#     # Прогоняем картинку
#     output, cam = generator(dummy_input)
#
#     print(generator.count_parameters())
#     print(cam.shape)
#     print(f"Вход: {dummy_input.shape}")
#     print(f"Выход: {output.shape}")
#     print(f"Мин. значение: {output.min().item():.4f}")
#     print(f"Макс. значение: {output.max().item():.4f}")