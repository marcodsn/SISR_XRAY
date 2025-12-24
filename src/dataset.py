import glob
import os

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class XRaySRDataset(Dataset):
    def __init__(
        self,
        root_dir,
        selected_folders,
        patch_size=256,
        scale_factor=4,
        test_mode=False,
    ):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.test_mode = test_mode
        self.image_files = []

        # Find images in selected folders
        for folder_name in selected_folders:
            search_pattern = os.path.join(root_dir, folder_name, "**", "*.png")
            found = glob.glob(search_pattern, recursive=True)
            if not found:
                search_pattern = os.path.join(root_dir, folder_name, "**", "*.jpg")
                found = glob.glob(search_pattern, recursive=True)
            self.image_files.extend(found)

        print(f"Found {len(self.image_files)} test images.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("L")

        # USE CENTER CROP FOR TESTING (Deterministic)
        # We crop first to ensure we have a valid HR ground truth
        if self.test_mode:
            hr_image = TF.center_crop(image, [self.patch_size, self.patch_size])
        else:
            crop_transform = transforms.RandomCrop(self.patch_size)
            hr_image = crop_transform(image)

        # Create LR input (Downscale -> Upscale)
        lr_size = (
            self.patch_size // self.scale_factor,
            self.patch_size // self.scale_factor,
        )

        # Downscale
        lr_image = TF.resize(
            hr_image, lr_size, interpolation=transforms.InterpolationMode.BICUBIC
        )

        # Upscale back to original size (so it matches UNet input dim)
        lr_upscaled = TF.resize(
            lr_image,
            (self.patch_size, self.patch_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        )

        to_tensor = transforms.ToTensor()
        return to_tensor(lr_upscaled), to_tensor(hr_image)
