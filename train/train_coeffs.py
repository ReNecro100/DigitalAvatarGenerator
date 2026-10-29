import torch

CARTOONS_DATASET_DIRECTORY = r"D:\Programming\Datasets\DigitalAvatars\animeee" #AnimeFaces
REAL_PHOTOS_DATASET_DIRECTORY = r"D:\Programming\Datasets\DigitalAvatars\thumbnails128x128" #FFHQ
EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.0002
BATCH_SIZE = 4

ADVERSIAL_WEIGHT = 1.0
CYCLE_WEIGHT = 5.0
IDENTITY_WEIGHT = 0.5