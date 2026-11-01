import torch
from torchvision import transforms
from PIL import Image
import os
import torchvision.utils as vutils
from train.models.generator import ResNetGenerator
from train.train_coeffs import DEVICE


def test_model(image_path, output_path=r"checkpoints\Alice 2.png"):
    # 1. Создаем генератор и загружаем веса
    model = ResNetGenerator().to(DEVICE)

    # Ищем файл весов (если сохранял с эпохами, берем последний)
    weights_path = "checkpoints/G_model_epoch30.pth"  # или G_epoch_10.pth
    if not os.path.exists(weights_path):
        print(f"Ошибка: Файл {weights_path} не найден!")
        return

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()  # Переводим в режим тестирования

    # 2. Подготовка картинки (те же трансформации, что и при обучении!)
    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = Image.open(image_path).convert("RGB")
    input_tensor = tf(img).unsqueeze(0).to(DEVICE)  # Добавляем размерность батча (1, 3, 128, 128)

    # 3. Генерация
    with torch.no_grad():
        output_tensor, _ = model(input_tensor)

    # 4. Сохранение результата (переводим из [-1, 1] обратно в [0, 1])
    vutils.save_image(output_tensor, output_path, normalize=True)
    print(f"✅ Результат сохранен в {output_path}")


if __name__ == "__main__":
    # Укажи путь к любому фото (например, из датасета FFHQ)
    test_image = r"checkpoints\Alice.png"
    test_model(test_image)