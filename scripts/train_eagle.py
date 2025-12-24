import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from external.Eagle_Loss.eagle_loss.eagle_loss import Eagle_Loss

# Local Imports
from src.dataset import XRaySRDataset
from src.models.diffusion import (
    ConditionalUNet,
    ResShiftSchedule,
    p_sample_loop_resshift,
)

# --- 1. Configuration & Hyperparameters ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Paths
DATA_ROOT = "/mnt/bigData/Downloads/chest_xrays"
RESULTS_PATH = "results/eagle"
os.makedirs(RESULTS_PATH, exist_ok=True)

# Training Config
BATCH_SIZE = 16
PATCH_SIZE = 256
SCALE = 4
NUM_EPOCHS = 40
LR = 1e-4

# Data Selection
SELECTED_FOLDERS = [
    "images_001",
    "images_002",
    "images_003",
    "images_004",
    "images_005",
    "images_006",
    "images_007",
]


# --- 2. Diffusion Specific Functions ---
def q_sample_resshift(x_0, y_0, t, schedule):
    """
    Adds noise to shift x_0 towards y_0 based on timestep t.
    Used during training to create noisy targets.
    """
    noise = torch.randn_like(x_0)
    eta_t = schedule.get_eta(t)
    sqrt_eta_t = torch.sqrt(eta_t)

    # The residual: e_0 = y_0 - x_0
    residual = y_0 - x_0

    # Mean: x_0 + eta_t * (y_0 - x_0)
    mean = x_0 + eta_t * residual

    # Std: kappa * sqrt(eta_t)
    std = schedule.kappa * sqrt_eta_t

    x_t = mean + std * noise
    return x_t


def save_validation_samples(model, schedule, loader, epoch, save_path):
    """
    Runs inference on a small batch and saves a comparison grid.
    """
    print(f"Generating validation samples for Epoch {epoch + 1}...")
    model.eval()

    # Get a mini-batch
    lr_imgs, hr_imgs = next(iter(loader))
    lr_imgs = lr_imgs[:3].to(DEVICE)
    hr_imgs = hr_imgs[:3].to(DEVICE)

    # Run Inference
    sr_imgs = p_sample_loop_resshift(model, lr_imgs, schedule)

    # Plotting
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        # Input
        axs[i, 0].imshow(lr_imgs[i].cpu().squeeze(), cmap="gray")
        axs[i, 0].set_title("Input (LR/Bicubic)")
        axs[i, 0].axis("off")

        # Output
        axs[i, 1].imshow(sr_imgs[i].cpu().squeeze(), cmap="gray")
        axs[i, 1].set_title("ResShift Eagle Output")
        axs[i, 1].axis("off")

        # Target
        axs[i, 2].imshow(hr_imgs[i].cpu().squeeze(), cmap="gray")
        axs[i, 2].set_title("Target (Original)")
        axs[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"eagle_epoch_{epoch + 1}.png"))
    plt.close()
    model.train()


# --- 3. Main Execution ---
if __name__ == "__main__":
    # A. Data Loading
    train_dataset = XRaySRDataset(
        root_dir=DATA_ROOT,
        patch_size=PATCH_SIZE,
        scale_factor=SCALE,
        selected_folders=SELECTED_FOLDERS,
        test_mode=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # B. Model & Schedule Setup
    schedule = ResShiftSchedule(num_steps=15, kappa=2.0, device=DEVICE)
    model = ConditionalUNet(in_channels=2, out_channels=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model Parameters: {param_count:.2f}M")

    # B.BIS Loss definition
    mse_criterion = (
        nn.MSELoss()
    )  # According to the paper, Eagle Loss shall be used in addition to MSE Loss
    eagle_criterion = Eagle_Loss(patch_size=3).to(DEVICE)
    eagle_weight = 1e-3

    # C. Training Loop
    print("Starting Training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for lr_imgs, hr_imgs in pbar:
            lr_imgs = lr_imgs.to(DEVICE)  # Condition (y_0)
            hr_imgs = hr_imgs.to(DEVICE)  # Target (x_0)

            batch_size = lr_imgs.shape[0]

            # 1. Sample Random Timesteps
            t = torch.randint(1, schedule.T + 1, (batch_size,), device=DEVICE)

            # 2. Add Noise (Forward Process)
            x_t = q_sample_resshift(hr_imgs, lr_imgs, t, schedule)

            # 3. Model Prediction
            pred_x0 = model(x_t, lr_imgs, t)

            # 4. Loss Calculation
            loss_mse = mse_criterion(pred_x0, hr_imgs)
            loss_eagle = eagle_criterion(pred_x0, hr_imgs)
            loss = loss_mse + (eagle_weight * loss_eagle)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.5f}")

        # Validation & Checkpointing
        save_validation_samples(model, schedule, train_loader, epoch, RESULTS_PATH)
        torch.save(model.state_dict(), os.path.join(RESULTS_PATH, "model_latest.pth"))

    print("Training Complete!")
