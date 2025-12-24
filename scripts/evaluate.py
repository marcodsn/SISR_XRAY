from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Import metrics from the utils file
from src.utils import (
    SSIM,
    EdgePSNR,
    FIDMetric,
    calculate_cnr,
    calculate_psnr,
    get_lpips_metric,
)


def read_image(path, device):
    """
    Reads an image as grayscale, normalizes to [0, 1], and returns both
    numpy (H, W) and tensor (1, 1, H, W) formats.
    """
    # Read as grayscale (1 channel)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    # Normalize to [0, 1]
    img_norm = img.astype(np.float32) / 255.0

    # Create Tensor: (Batch=1, Channel=1, H, W)
    img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)

    return img_norm, img_tensor


def evaluate_datasets():
    # --- Configuration ---
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running evaluation on device: {device}")

    # Paths
    gt_dir = Path("data/HR/patches")
    lr_dir = Path("data/LR")
    sr_root_dir = Path("data/SR")
    output_dir = Path("results/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Metrics that need persistence
    edge_psnr_metric = EdgePSNR(device=device)
    lpips_metric = get_lpips_metric(net="alex", device=device)

    # Identify all method directories to evaluate
    # We include LR as a method, plus any subfolders in SR (e.g., diffusion, eagle, unet)
    methods = {"Bicubic_LR": lr_dir}

    if sr_root_dir.exists():
        for d in sr_root_dir.iterdir():
            if d.is_dir():
                methods[d.name] = d

    # Dataframes to store results
    detailed_results = []
    summary_results = []

    # --- Evaluation Loop ---
    for method_name, method_path in methods.items():
        print(f"\n--- Evaluating Method: {method_name} ---")

        # Initialize FID for this specific distribution/method
        fid_metric = FIDMetric(device=device)

        # Get list of GT files
        gt_files = sorted(list(gt_dir.glob("*.png")) + list(gt_dir.glob("*.jpg")))

        if not gt_files:
            print(f"Warning: No images found in {gt_dir}")
            continue

        # Accumulators for mean calculation
        method_metrics = {
            "PSNR": [],
            "SSIM": [],
            "CNR": [],
            "EdgePSNR": [],
            "LPIPS": [],
        }

        for gt_path in tqdm(gt_files, desc=f"Processing {method_name}"):
            filename = gt_path.name
            pred_path = method_path / filename

            if not pred_path.exists():
                # Try checking extensions if they differ (optional, but good for safety)
                continue

            # Load Images
            # gt_np: (H, W) [0,1], gt_t: (1, 1, H, W) [0,1]
            gt_np, gt_t = read_image(str(gt_path), device)
            pred_np, pred_t = read_image(str(pred_path), device)

            # --- Calculate Per-Image Metrics ---

            # 1. PSNR (Expects [0,1])
            psnr_val = calculate_psnr(pred_t, gt_t, data_range=1.0).item()

            # 2. SSIM (Expects Numpy (H,W) or (H,W,C))
            # We add a dummy channel axis for consistency with utils signature if needed,
            # though skimage often handles 2D. Utils wrapper uses channel_axis=2.
            ssim_val = SSIM(
                pred_np[:, :, np.newaxis],
                gt_np[:, :, np.newaxis],
                data_range=1.0,
                channel_axis=2,
            )

            # 3. CNR (Contrast to Noise Ratio)
            # Calculated on the Prediction image alone (usually) to check contrast preservation
            cnr_val = calculate_cnr(pred_t)

            # 4. EdgePSNR
            edge_psnr_val = edge_psnr_metric(pred_t, gt_t)

            # 5. LPIPS
            # LPIPS expects input in range [-1, 1] and 3 channels.
            # We repeat the grayscale channel to 3.
            pred_t_norm = (pred_t - 0.5) * 2.0
            gt_t_norm = (gt_t - 0.5) * 2.0

            pred_t_rgb = pred_t_norm.repeat(1, 3, 1, 1)
            gt_t_rgb = gt_t_norm.repeat(1, 3, 1, 1)

            lpips_val = lpips_metric(pred_t_rgb, gt_t_rgb).item()

            # 6. Accumulate FID features
            # FIDMetric handles 1->3 channel conversion internally
            fid_metric.update(gt_t, pred_t)

            # Store per-image details
            row = {
                "Filename": filename,
                "Method": method_name,
                "PSNR": psnr_val,
                "SSIM": ssim_val,
                "CNR": cnr_val,
                "EdgePSNR": edge_psnr_val,
                "LPIPS": lpips_val,
            }
            detailed_results.append(row)

            # Accumulate for local mean
            method_metrics["PSNR"].append(psnr_val)
            method_metrics["SSIM"].append(ssim_val)
            method_metrics["CNR"].append(cnr_val)
            method_metrics["EdgePSNR"].append(edge_psnr_val)
            method_metrics["LPIPS"].append(lpips_val)

        # --- Compute FID (Dataset level) ---
        print("Computing FID...")
        try:
            fid_score = fid_metric.compute()
        except Exception as e:
            print(f"FID computation failed for {method_name}: {e}")
            fid_score = np.nan

        # --- Store Summary ---
        summary_row = {
            "Method": method_name,
            "FID": fid_score,
            "Mean_PSNR": np.mean(method_metrics["PSNR"]),
            "Mean_SSIM": np.mean(method_metrics["SSIM"]),
            "Mean_CNR": np.mean(method_metrics["CNR"]),
            "Mean_EdgePSNR": np.mean(method_metrics["EdgePSNR"]),
            "Mean_LPIPS": np.mean(method_metrics["LPIPS"]),
        }
        summary_results.append(summary_row)
        print(
            f"Finished {method_name}: PSNR={summary_row['Mean_PSNR']:.2f}, FID={fid_score:.4f}"
        )

    # --- Save Results ---
    df_detailed = pd.DataFrame(detailed_results)
    df_summary = pd.DataFrame(summary_results)

    detailed_csv_path = output_dir / "detailed_metrics.csv"
    summary_csv_path = output_dir / "summary_metrics.csv"

    df_detailed.to_csv(detailed_csv_path, index=False)
    df_summary.to_csv(summary_csv_path, index=False)

    print("\nEvaluation Complete!")
    print(f"Detailed results saved to: {detailed_csv_path}")
    print(f"Summary results saved to: {summary_csv_path}")
    print("\nSummary Table:")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    evaluate_datasets()
