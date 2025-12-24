import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
from torchvision.models import Inception_V3_Weights, inception_v3


def calculate_cnr(img_tensor, roi_percentile=85, bg_percentile=15):
    """
    Automated CNR calculation using intensity percentiles.
    """
    if torch.is_tensor(img_tensor):
        img_np = img_tensor.detach().cpu().numpy().flatten()
    else:
        img_np = img_tensor.flatten()

    roi_thresh = np.percentile(img_np, roi_percentile)
    bg_thresh = np.percentile(img_np, bg_percentile)

    roi_mask = img_np >= roi_thresh
    bg_mask = img_np <= bg_thresh

    if np.sum(roi_mask) == 0 or np.sum(bg_mask) == 0:
        return 0.0

    mu_roi = np.mean(img_np[roi_mask])
    mu_bg = np.mean(img_np[bg_mask])
    sigma_bg = np.std(img_np[bg_mask])

    if sigma_bg < 1e-6:
        return 0.0

    cnr = np.abs(mu_roi - mu_bg) / sigma_bg
    return cnr


def calculate_psnr(img1, img2, data_range=1.0):
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR).
    Args:
        img1, img2: Tensors (B, C, H, W) or (C, H, W).
        data_range: Dynamic range of the images (usually 1.0 or 255.0).
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(torch.tensor(data_range)) - 10 * torch.log10(mse)


def SSIM(img1, img2, data_range=1.0, channel_axis=2):
    return ssim(img1, img2, data_range=data_range, channel_axis=channel_axis)


class EdgePSNR(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        # Define Sobel Kernels for X and Y directions
        self.kx = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            .view(1, 1, 3, 3)
            .to(device)
        )
        self.ky = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            .view(1, 1, 3, 3)
            .to(device)
        )

    def get_edge_map(self, img):
        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        edge_x = F.conv2d(img, self.kx, padding=1)
        edge_y = F.conv2d(img, self.ky, padding=1)
        edge_mag = torch.sqrt(edge_x**2 + edge_y**2)
        return edge_mag

    def forward(self, pred, target):
        if pred.shape[1] == 3:
            pred = TF.rgb_to_grayscale(pred)
            target = TF.rgb_to_grayscale(target)

        pred_edges = self.get_edge_map(pred)
        target_edges = self.get_edge_map(target)

        mse = torch.mean((pred_edges - target_edges) ** 2)
        if mse == 0:
            return 100.0

        return 10 * torch.log10(1.0 / mse).item()


def get_lpips_metric(net="alex", device="cuda"):
    """Returns the LPIPS loss function."""
    return lpips.LPIPS(net=net).to(device)


class FIDMetric:
    """Computes FID between two distributions."""

    def __init__(self, device="cuda"):
        self.device = device
        self.inception = inception_v3(
            weights=Inception_V3_Weights.DEFAULT, transform_input=False
        ).to(device)
        self.inception.fc = nn.Identity()
        self.inception.eval()
        self.real_features = []
        self.fake_features = []

    @torch.no_grad()
    def update(self, real_imgs, fake_imgs):
        real_imgs = F.interpolate(
            real_imgs, size=(299, 299), mode="bilinear", align_corners=False
        )
        fake_imgs = F.interpolate(
            fake_imgs, size=(299, 299), mode="bilinear", align_corners=False
        )

        # Normalize to [-1, 1]
        real_imgs = (real_imgs - 0.5) / 0.5
        fake_imgs = (fake_imgs - 0.5) / 0.5

        if real_imgs.shape[1] == 1:
            real_imgs = real_imgs.repeat(1, 3, 1, 1)
            fake_imgs = fake_imgs.repeat(1, 3, 1, 1)

        feat_real = self.inception(real_imgs)
        feat_fake = self.inception(fake_imgs)

        self.real_features.append(feat_real.cpu().numpy())
        self.fake_features.append(feat_fake.cpu().numpy())

    def compute(self):
        act1 = np.concatenate(self.real_features, axis=0)
        act2 = np.concatenate(self.fake_features, axis=0)

        mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

        ssdiff = np.sum((mu1 - mu2) ** 2)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid


if __name__ == "__main__":
    device = "cpu"  # Change to "cuda" if available

    # --- Test CNR ---
    test_img = np.random.rand(256, 256) * 255
    cnr_value = calculate_cnr(test_img)
    print(f"CNR Value: {cnr_value:.4f}")

    # Setup dummy images for tensor metrics
    # Shape: (Batch, Channels, Height, Width)
    img1 = torch.rand(1, 3, 256, 256).to(device)
    img2 = torch.rand(1, 3, 256, 256).to(device)

    # --- Test PSNR ---
    # Standard PSNR assumes inputs are identical scale (here [0, 1])
    psnr_val = calculate_psnr(img1, img2, data_range=1.0)
    print(f"PSNR Value: {psnr_val:.4f} dB")

    # --- Test SSIM ---
    # SSIM expects numpy arrays in (H, W, C) format
    img1_np = img1.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img2_np = img2.squeeze(0).permute(1, 2, 0).cpu().numpy()
    ssim_val = SSIM(img1_np, img2_np, data_range=1.0, channel_axis=2)
    print(f"SSIM Value: {ssim_val:.4f}")

    # --- Test EdgePSNR ---
    edge_psnr = EdgePSNR(device=device)
    # EdgePSNR handles conversion to grayscale internally if needed
    edge_psnr_val = edge_psnr(img1, img2)
    print(f"Edge PSNR Value: {edge_psnr_val:.4f} dB")

    # --- Test LPIPS ---
    # LPIPS expects inputs roughly in [-1, 1]
    lpips_metric = get_lpips_metric(net="alex", device=device)
    img1_norm = (img1 - 0.5) * 2
    img2_norm = (img2 - 0.5) * 2
    lpips_val = lpips_metric(img1_norm, img2_norm)
    print(f"LPIPS Value: {lpips_val.item():.4f}")

    # --- Test FID ---
    fid_metric = FIDMetric(device=device)
    for _ in range(2):
        real_batch = torch.rand(4, 3, 64, 64).to(device)
        fake_batch = torch.rand(4, 3, 64, 64).to(device)
        fid_metric.update(real_batch, fake_batch)
    fid_val = fid_metric.compute()
    print(f"FID Value: {fid_val:.4f}")
