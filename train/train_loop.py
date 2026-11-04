import torch
import torch.nn as nn
import itertools
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from data.datasets.dired_dataset import DiredDataset
from train.models.discriminator import Discriminator
from train.models.generator import ResnetGenerator
from train.models.AdaILN import RhoClipper
from train_coeffs import *

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # 1. Подготовка данных
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cartoon_dataset = DiredDataset(CARTOONS_DATASET_DIRECTORY, tf)
    real_photos_dataset = DiredDataset(REAL_PHOTOS_DATASET_DIRECTORY, tf)

    # В оригинале DataLoader без shuffle, но мы оставим shuffle=True для лучшего перемешивания
    cartoon_dataloader = DataLoader(cartoon_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                    pin_memory=True)
    real_photos_dataloader = DataLoader(real_photos_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                        pin_memory=True)

    # 2. Создание моделей (строго по репозиторию)
    # Генераторы (A->B и B->A)
    genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=N_RES, img_size=IMG_SIZE, light=LIGHT).to(DEVICE)
    genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=N_RES, img_size=IMG_SIZE, light=LIGHT).to(DEVICE)

    # Дискриминаторы (2 глобальных, 2 локальных)
    disGA = Discriminator(input_nc=3, ndf=64, n_layers=7).to(DEVICE)
    disGB = Discriminator(input_nc=3, ndf=64, n_layers=7).to(DEVICE)
    disLA = Discriminator(input_nc=3, ndf=64, n_layers=5).to(DEVICE)
    disLB = Discriminator(input_nc=3, ndf=64, n_layers=5).to(DEVICE)

    # 3. Функции потерь
    L1_loss = nn.L1Loss().to(DEVICE)
    MSE_loss = nn.MSELoss().to(DEVICE)
    BCE_loss = nn.BCEWithLogitsLoss().to(DEVICE)

    # 4. Оптимизаторы (один для всех G, один для всех D)
    G_optim = torch.optim.Adam(itertools.chain(genA2B.parameters(), genB2A.parameters()), lr=LEARNING_RATE,
                               betas=(0.5, 0.999))
    D_optim = torch.optim.Adam(
        itertools.chain(disGA.parameters(), disGB.parameters(), disLA.parameters(), disLB.parameters()),
        lr=LEARNING_RATE, betas=(0.5, 0.999))

    # 5. RhoClipper (ограничивает значения rho в adaILN от 0 до 1)
    Rho_clipper = RhoClipper(0, 1)

    # Директория для чекпоинтов
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # 6. Цикл обучения (один общий цикл, как в оригинале)
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}:")

        # Создаем бесконечные итераторы для батчей
        loader_X = iter(real_photos_dataloader)
        loader_Y = iter(cartoon_dataloader)
        max_batches = 20 #min(len(real_photos_dataloader), len(cartoon_dataloader))

        for _ in tqdm(range(max_batches)):
            # Получаем батчи (real_A = селфи, real_B = аниме)
            try:
                real_A = next(loader_X)
            except StopIteration:
                loader_X = iter(real_photos_dataloader)
                real_A = next(loader_X)

            try:
                real_B = next(loader_Y)
            except StopIteration:
                loader_Y = iter(cartoon_dataloader)
                real_B = next(loader_Y)

            real_A = real_A.to(DEVICE)
            real_B = real_B.to(DEVICE)

            # ==========================================
            # ОБНОВЛЕНИЕ ДИСКРИМИНАТОРОВ
            # ==========================================
            D_optim.zero_grad()

            # Генерируем фейки (без градиентов для G)
            with torch.no_grad():
                fake_A2B, _, _ = genA2B(real_A)
                fake_B2A, _, _ = genB2A(real_B)

            # Прогон через дискриминаторы (3 выхода: out, cam_logit, heatmap)
            real_GA_logit, real_GA_cam_logit, _ = disGA(real_A)
            real_LA_logit, real_LA_cam_logit, _ = disLA(real_A)
            real_GB_logit, real_GB_cam_logit, _ = disGB(real_B)
            real_LB_logit, real_LB_cam_logit, _ = disLB(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = disLB(fake_A2B)

            # Лоссы дискриминаторов (MSE для adversarial и CAM)
            D_ad_loss_GA = MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit)) + MSE_loss(fake_GA_logit,
                                                                                              torch.zeros_like(
                                                                                                  fake_GA_logit))
            D_ad_cam_loss_GA = MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit)) + MSE_loss(
                fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit))

            D_ad_loss_LA = MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit)) + MSE_loss(fake_LA_logit,
                                                                                              torch.zeros_like(
                                                                                                  fake_LA_logit))
            D_ad_cam_loss_LA = MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit)) + MSE_loss(
                fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit))

            D_ad_loss_GB = MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit)) + MSE_loss(fake_GB_logit,
                                                                                              torch.zeros_like(
                                                                                                  fake_GB_logit))
            D_ad_cam_loss_GB = MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit)) + MSE_loss(
                fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit))

            D_ad_loss_LB = MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit)) + MSE_loss(fake_LB_logit,
                                                                                              torch.zeros_like(
                                                                                                  fake_LB_logit))
            D_ad_cam_loss_LB = MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit)) + MSE_loss(
                fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit))

            D_loss_A = ADVERSIAL_WEIGHT * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
            D_loss_B = ADVERSIAL_WEIGHT * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

            Discriminator_loss = D_loss_A + D_loss_B
            Discriminator_loss.backward()
            D_optim.step()

            # ==========================================
            # ОБНОВЛЕНИЕ ГЕНЕРАТОРОВ
            # ==========================================
            G_optim.zero_grad()

            # Генерируем фейки (теперь с градиентами)
            fake_A2B, fake_A2B_cam_logit, _ = genA2B(real_A)
            fake_B2A, fake_B2A_cam_logit, _ = genB2A(real_B)

            # Цикл (возврат к оригиналу)
            fake_A2B2A, _, _ = genB2A(fake_A2B)
            fake_B2A2B, _, _ = genA2B(fake_B2A)

            # Identity (генератор не должен менять то, что уже в его домене)
            fake_A2A, fake_A2A_cam_logit, _ = genB2A(real_A)
            fake_B2B, fake_B2B_cam_logit, _ = genA2B(real_B)

            # Прогон через дискриминаторы (для лоссов G)
            fake_GA_logit, fake_GA_cam_logit, _ = disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = disLB(fake_A2B)

            # Adversarial лоссы генераторов (MSE)
            G_ad_loss_GA = MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit))
            G_ad_cam_loss_GA = MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit))
            G_ad_loss_LA = MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit))
            G_ad_cam_loss_LA = MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit))
            G_ad_loss_GB = MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit))
            G_ad_cam_loss_GB = MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit))
            G_ad_loss_LB = MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit))
            G_ad_cam_loss_LB = MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit))

            # Cycle и Identity лоссы (L1)
            G_recon_loss_A = L1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = L1_loss(fake_B2A2B, real_B)
            G_identity_loss_A = L1_loss(fake_A2A, real_A)
            G_identity_loss_B = L1_loss(fake_B2B, real_B)

            # CAM лоссы генераторов (BCE!)
            G_cam_loss_A = BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit)) + BCE_loss(
                fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit))
            G_cam_loss_B = BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit)) + BCE_loss(
                fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit))

            # Суммарные лоссы G
            G_loss_A = ADVERSIAL_WEIGHT * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + \
                       CYCLE_WEIGHT * G_recon_loss_A + \
                       IDENTITY_WEIGHT * G_identity_loss_A + \
                       CAM_WEIGHT * G_cam_loss_A

            G_loss_B = ADVERSIAL_WEIGHT * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + \
                       CYCLE_WEIGHT * G_recon_loss_B + \
                       IDENTITY_WEIGHT * G_identity_loss_B + \
                       CAM_WEIGHT * G_cam_loss_B

            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()
            G_optim.step()

            # Обязательно после шага оптимизатора ограничиваем rho в adaILN
            genA2B.apply(Rho_clipper)
            genB2A.apply(Rho_clipper)

        # Сохранение
        if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
            torch.save({
                'genA2B': genA2B.state_dict(),
                'genB2A': genB2A.state_dict(),
                'disGA': disGA.state_dict(),
                'disGB': disGB.state_dict(),
                'disLA': disLA.state_dict(),
                'disLB': disLB.state_dict()
            }, os.path.join(checkpoints_dir, f"checkpoint_epoch_{epoch + 1}.pth"))
            print(f"✅ Эпоха {epoch + 1} сохранена!")