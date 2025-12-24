import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local Imports
from src.dataset import XRaySRDataset
from src.models.unet import UNetSR

# --- 1. Configuration & Hyperparameters ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Paths
DATA_ROOT = "/mnt/bigData/Downloads/chest_xrays"
RESULTS_PATH = "results/unet"
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


# --- 2. Helper Functions ---
def save_validation_samples(model, loader, epoch, save_path):
    """
    Runs inference on a small batch and saves a comparison grid.
    """
    print(f"Generating validation samples for Epoch {epoch + 1}...")
    model.eval()

    with torch.no_grad():
        # Get a mini-batch
        lr_imgs, hr_imgs = next(iter(loader))
        lr_imgs = lr_imgs[:3].to(DEVICE)
        hr_imgs = hr_imgs[:3].to(DEVICE)

        # Run Inference (Direct Forward Pass)
        sr_imgs = model(lr_imgs)

    # Plotting
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    for i in range(3):
        # Input
        axs[i, 0].imshow(lr_imgs[i].cpu().squeeze(), cmap="gray")
        axs[i, 0].set_title("Input (LR/Bicubic)")
        axs[i, 0].axis("off")

        # Output
        axs[i, 1].imshow(sr_imgs[i].cpu().squeeze(), cmap="gray")
        axs[i, 1].set_title("UNet Output")
        axs[i, 1].axis("off")

        # Target
        axs[i, 2].imshow(hr_imgs[i].cpu().squeeze(), cmap="gray")
        axs[i, 2].set_title("Target (Original)")
        axs[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"unet_epoch_{epoch + 1}.png"))
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
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # B. Model Setup
    model = UNetSR(n_channels=1, n_classes=1).to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Model Parameters: {param_count:.2f}M")

    # C. Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.L1Loss()

    # D. Training Loop
    print("Starting Training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for lr_imgs, hr_imgs in pbar:
            lr_imgs = lr_imgs.to(DEVICE)
            hr_imgs = hr_imgs.to(DEVICE)

            # Forward pass
            outputs = model(lr_imgs)

            # Loss calculation
            loss = criterion(outputs, hr_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.5f}")

        # Validation & Checkpointing
        save_validation_samples(model, train_loader, epoch, RESULTS_PATH)
        torch.save(model.state_dict(), os.path.join(RESULTS_PATH, "model_latest.pth"))

    print("Training Complete!")
