import torch

CARTOONS_DATASET_DIRECTORY = r"D:\Programming\Datasets\DigitalAvatars\cartoonset100k_jpg" #CartoonSet
REAL_PHOTOS_DATASET_DIRECTORY = r"D:\Programming\Datasets\DigitalAvatars\thumbnails128x128" #FFHQ
EPOCHS = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.0002
BATCH_SIZE = 4