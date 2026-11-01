from torch import nn
import itertools
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from data.datasets.double_dired_dataset import DoubledDiredDataset
from data.datasets.dired_dataset import DiredDataset

from train.models.discriminator import PatchGANDiscriminator
from train.models.generator import ResNetGenerator

from train_coeffs import *

if __name__ == '__main__':
    #Creating datasets
    tf = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    cartoon_dataset = DiredDataset(CARTOONS_DATASET_DIRECTORY,tf)
    real_photos_dataset = DiredDataset(REAL_PHOTOS_DATASET_DIRECTORY,tf)

    #Creating dataloaders
    cartoon_dataloader = DataLoader(cartoon_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    real_photos_dataloader = DataLoader(real_photos_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    #creating models

    generator_g = ResNetGenerator().to(DEVICE) #photos->cartoons
    optimizer_g = torch.optim.Adam(generator_g.parameters(), lr=LEARNING_RATE)

    generator_f = ResNetGenerator().to(DEVICE) #cartoons->photos
    optimizer_f = torch.optim.Adam(generator_f.parameters(), lr=LEARNING_RATE)

    discriminator_y = PatchGANDiscriminator().to(DEVICE) #cartoons expert
    criterion_d_y = nn.MSELoss()
    optimizer_y = torch.optim.Adam(discriminator_y.parameters(), lr=LEARNING_RATE)

    discriminator_x = PatchGANDiscriminator().to(DEVICE) #photos expert
    criterion_d_x = nn.MSELoss()
    optimizer_x = torch.optim.Adam(discriminator_x.parameters(), lr=LEARNING_RATE)

    print(DEVICE)

    # dir for checkpoints
    current_dir = os.path.dirname(os.path.abspath(__file__))  # папка train
    project_root = os.path.dirname(current_dir)  # корень проекта
    checkpoints_dir = os.path.join(project_root, "checkpoints")

    os.makedirs(checkpoints_dir, exist_ok=True)

    #train loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}:")
        print(f"Training discriminators...")
        # training discriminators
        generator_g.eval()
        generator_f.eval()

        loader_X = iter(real_photos_dataloader)
        loader_Y = iter(cartoon_dataloader)
        max_batches = min(len(real_photos_dataloader), len(cartoon_dataloader))

        for _ in tqdm(range(max_batches)):
            # batches
            real_X = next(loader_X)
            real_Y = next(loader_Y)

            real_X = real_X.to(DEVICE)
            real_Y = real_Y.to(DEVICE)

            # true photos
            pred_real_X, cam_D_X_real = discriminator_x(real_X)
            loss_D_X_real = criterion_d_x(pred_real_X, torch.ones_like(pred_real_X))
            loss_cam_D_X_real = criterion_d_x(cam_D_X_real, torch.ones_like(cam_D_X_real))

            # fake photos
            fake_X, _ = generator_f(real_Y)
            pred_fake_X, cam_D_X_fake = discriminator_x(fake_X.detach())
            loss_D_X_fake = criterion_d_x(pred_fake_X, torch.zeros_like(pred_fake_X))
            loss_cam_D_X_fake = criterion_d_x(cam_D_X_fake, torch.zeros_like(cam_D_X_fake))

            loss_cam_D_X = (loss_cam_D_X_real + loss_cam_D_X_fake) * 0.5
            loss_D_X = (loss_D_X_real + loss_D_X_fake) * 0.5 + loss_cam_D_X

            # real cartoons
            pred_real_Y, cam_D_Y_real = discriminator_y(real_Y)
            loss_D_Y_real = criterion_d_y(pred_real_Y, torch.ones_like(pred_real_Y))
            loss_cam_D_Y_real = criterion_d_y(cam_D_Y_real, torch.ones_like(cam_D_Y_real))

            # fake cartoons
            fake_Y, _ = generator_g(real_X)
            pred_fake_Y, cam_D_Y_fake = discriminator_y(fake_Y.detach())
            loss_D_Y_fake = criterion_d_y(pred_fake_Y, torch.zeros_like(pred_fake_Y))
            loss_cam_D_Y_fake = criterion_d_y(cam_D_Y_fake, torch.zeros_like(cam_D_Y_fake))

            loss_cam_D_Y = (loss_cam_D_Y_real + loss_cam_D_Y_fake) * 0.5
            loss_D_Y = (loss_D_Y_real + loss_D_Y_fake) * 0.5 + loss_cam_D_Y

            # update the weights
            optimizer_x.zero_grad()
            optimizer_y.zero_grad()

            loss_D_X.backward()
            loss_D_Y.backward()

            optimizer_x.step()
            optimizer_y.step()

        print(f"Training generators...")
        # training generators
        generator_g.train()
        generator_f.train()

        discriminator_y.eval()
        discriminator_x.eval()

        loader_X = iter(real_photos_dataloader)
        loader_Y = iter(cartoon_dataloader)

        torch.cuda.empty_cache()
        for _ in tqdm(range(max_batches)):
            # batches
            real_X = next(loader_X)
            real_Y = next(loader_Y)

            real_X = real_X.to(DEVICE)
            real_Y = real_Y.to(DEVICE)

            # G generates
            cartoon_G, cam_G = generator_g(real_X)

            # adversarial loss
            with torch.no_grad():
                pred_fake_Y, cam_D_Y = discriminator_y(cartoon_G)
            loss_adversarial_G = criterion_d_y(pred_fake_Y, torch.ones_like(pred_fake_Y))

            # cycle loss
            rec_X, _ = generator_f(cartoon_G)
            loss_cycle_X = torch.nn.functional.l1_loss(rec_X, real_X)

            # identity loss
            identity_Y, _ = generator_g(real_Y)
            loss_identity_Y = torch.nn.functional.l1_loss(identity_Y, real_Y)

            # CAM Loss для генератора: хотим, чтобы cam_G была похожа на cam_D_Y
            loss_cam_G = torch.nn.functional.l1_loss(
                torch.nn.functional.adaptive_avg_pool2d(cam_G, (1,1)).mean(dim=1, keepdim=True),
                torch.nn.functional.adaptive_avg_pool2d(cam_D_Y, (1,1)).mean(dim=1, keepdim=True)
            )

            # total loss
            loss_G = (ADVERSIAL_WEIGHT * loss_adversarial_G) + (CYCLE_WEIGHT * loss_cycle_X) + (IDENTITY_WEIGHT * loss_identity_Y) + (CAM_WEIGHT * loss_cam_G)



            # F generates
            cartoon_F, cam_F = generator_f(real_Y)

            # adversarial loss
            with torch.no_grad():
                pred_fake_X, cam_D_X = discriminator_x(cartoon_F)
            loss_adversarial_F = criterion_d_x(pred_fake_X, torch.ones_like(pred_fake_X))

            # cycle loss
            rec_Y, _ = generator_g(cartoon_F)
            loss_cycle_Y = torch.nn.functional.l1_loss(rec_Y, real_Y)

            # identity loss
            identity_X, _ = generator_f(real_X)
            loss_identity_X = torch.nn.functional.l1_loss(identity_X, real_X)

            # CAM Loss для генератора: хотим, чтобы cam_F была похожа на cam_D_X
            loss_cam_F = torch.nn.functional.l1_loss(
                torch.nn.functional.adaptive_avg_pool2d(cam_F, (1,1)).mean(dim=1, keepdim=True),
                torch.nn.functional.adaptive_avg_pool2d(cam_D_X, (1,1)).mean(dim=1, keepdim=True)
            )

            # (lambda_cycle=10, lambda_identity=0.5)
            loss_F = (ADVERSIAL_WEIGHT * loss_adversarial_F) + (CYCLE_WEIGHT * loss_cycle_Y) + (IDENTITY_WEIGHT * loss_identity_X) + (CAM_WEIGHT * loss_cam_F)



            # update the weights
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()

            loss_G.backward()
            loss_F.backward()

            optimizer_g.step()
            optimizer_f.step()
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            # Сохраняем веса
            torch.save(generator_g.state_dict(), os.path.join(checkpoints_dir, f"G_model_epoch{epoch + 1}.pth"))
            torch.save(generator_f.state_dict(), os.path.join(checkpoints_dir, f"F_model_epoch{epoch + 1}.pth"))
            torch.save(discriminator_x.state_dict(), os.path.join(checkpoints_dir, f"D_X_model_epoch{epoch + 1}.pth"))
            torch.save(discriminator_y.state_dict(), os.path.join(checkpoints_dir, f"D_Y_model_epoch{epoch + 1}.pth"))
