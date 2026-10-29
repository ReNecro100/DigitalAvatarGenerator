from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os

class DiredDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        for img in os.listdir(root):
            if img.lower().endswith(".png") or img.lower().endswith(".jpg"):
                self.images.append(os.path.join(root, img))

    def __getitem__(self, index):
        img_path = self.images[index]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image
    def __len__(self):
        return len(self.images)


# if __name__ == "__main__":
#     tf = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#
#     # Инициализируем датасет
#     datasets = RealPhotosDataset(transform=tf)
#
#     print(f"Всего картинок: {len(datasets)}")
#
#     # Берем первую картинку
#     sample = datasets[0]
#     print(f"Размер тензора: {sample.shape}")  # Должно быть torch.Size([3, 256, 256])
#     print(f"Мин. значение: {sample.min():.2f}, Макс. значение: {sample.max():.2f}")  # От -1 до 1