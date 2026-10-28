import torch
from torchvision import transforms
from PIL import Image
from train.models.generator import ResNetGenerator
from train.train_coeffs import DEVICE

model = ResNetGenerator().to(DEVICE)
model.load_state_dict(torch.load("checkpoints/G_model.pth", map_location=DEVICE))
model.eval()

tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Загружаем твое фото (замени путь на свой)
img = Image.open(r"D:\Programming\Datasets\DigitalAvatars\Alan Wilder.png").convert("RGB")
input_tensor = tf(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = model(input_tensor)

print("Мин. значение:", output.min().item())
print("Макс. значение:", output.max().item())
print("Среднее значение:", output.mean().item())