import torch

# --- Пути к данным ---
# trainA (селфи) -> REAL_PHOTOS_DATASET_DIRECTORY
# trainB (аниме) -> CARTOONS_DATASET_DIRECTORY
CARTOONS_DATASET_DIRECTORY = r"D:\Programming\Datasets\DigitalAvatars\selfie2anime\trainB"
REAL_PHOTOS_DATASET_DIRECTORY = r"D:\Programming\Datasets\DigitalAvatars\selfie2anime\trainA"

# --- Общие настройки ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.0002
BATCH_SIZE = 1          # Если вылетит CUDA out of memory (из-за 4 дискриминаторов), ставь 1 или 2
EPOCHS = 50

# --- Параметры архитектуры ---
IMG_SIZE = 128          # Должно совпадать с transforms.Resize в train_loop.py
LIGHT = False           # False = полная архитектура (как в статье)
N_RES = 6               # Количество residual блоков (для 128x128 обычно 6)

# --- Веса лоссов (строго из статьи/репозитория) ---
ADVERSIAL_WEIGHT = 1.0
CYCLE_WEIGHT = 10.0
IDENTITY_WEIGHT = 10.0
CAM_WEIGHT = 1000.0     # Строго 1000! Так как CAM-лосс для G теперь считается через BCEWithLogitsLoss