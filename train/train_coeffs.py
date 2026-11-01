import torch

#selfie2anime
CARTOONS_DATASET_DIRECTORY = r"D:\Programming\Datasets\DigitalAvatars\selfie2anime\trainB"
REAL_PHOTOS_DATASET_DIRECTORY = r"D:\Programming\Datasets\DigitalAvatars\selfie2anime\trainA"
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.0002
BATCH_SIZE = 4

ADVERSIAL_WEIGHT = 1.0
CYCLE_WEIGHT = 10.0
IDENTITY_WEIGHT = 10.0
CAM_WEIGHT = 1.0