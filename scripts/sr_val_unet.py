## This script generates the SR images used for evaluation of the unet model
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

# Local Imports
from src.models.unet import UNetSR

# --- 1. Configuration & Hyperparameters ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

WEIGHTS_PATH = "results/unet/model_latest.pth"
INPUT_DIR = "data/LR/"
OUTPUT_DIR = "data/SR/unet"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model Config
BATCH_SIZE = 8


# --- 2. Inference Specific Functions & Classes ---
class InferenceDataset(Dataset):
    """
    Simple Dataset to load images from a directory for batch processing.
    """

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = [
            f
            for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        path = os.path.join(self.root_dir, filename)

        # Load Grayscale
        img = Image.open(path).convert("L")

        img_tensor = TF.to_tensor(img)

        return img_tensor, filename


def load_model(weights_path, device):
    """
    Initializes the UNetSR architecture and loads weights.
    """
    # Initialize Model (Grayscale: n_channels=1, n_classes=1)
    model = UNetSR(n_channels=1, n_classes=1, bilinear=True).to(device)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found at: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)

    # Handle state_dict wrapper if present
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle 'module.' prefix from DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def save_batch(tensors, filenames, output_dir):
    """
    Saves a batch of tensors to disk.
    """
    for i, filename in enumerate(filenames):
        # Clamp to [0, 1] to ensure valid image range
        tensor = tensors[i].clamp(0, 1).cpu()
        result_img = TF.to_pil_image(tensor)

        save_path = os.path.join(output_dir, filename)
        result_img.save(save_path)


# --- 3. Main Execution ---
if __name__ == "__main__":
    # A. Load Model
    print("Initializing Inference...")
    try:
        model = load_model(WEIGHTS_PATH, DEVICE)
        print(f"Model loaded from {WEIGHTS_PATH}")
    except Exception as e:
        print(f"Setup Error: {e}")
        exit()

    # B. Prepare Data Loader
    val_dataset = InferenceDataset(INPUT_DIR)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Found {len(val_dataset)} images. Processing in batches of {BATCH_SIZE}...")

    # C. Run Inference Loop
    with torch.no_grad():
        for batch_imgs, filenames in tqdm(val_loader, desc="Processing"):
            # 1. Move to device
            input_tensor = batch_imgs.to(DEVICE)

            # 2. Forward Pass (Single Step)
            # The model internally handles "input + residual"
            sr_output = model(input_tensor)

            # 3. Save Results
            save_batch(sr_output, filenames, OUTPUT_DIR)

    print(f"Inference Complete! Results saved to {OUTPUT_DIR}")
