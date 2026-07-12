from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from prepare_and_preprocess_data import HeatmapDataset
from LightweightAutoencoder import LightweightAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightweightAutoencoder().to(device)
model.eval()

project_root = Path(__file__).resolve().parents[2]
heatmap_dir = project_root / "heatmaps" / "patched" / "d9fda4e4-0b76e3f0"

if not heatmap_dir.exists():
    raise FileNotFoundError(f"Heatmap folder not found: {heatmap_dir}")

# Load one heatmap sample
patched_dataset = HeatmapDataset(folder_path=str(heatmap_dir))
test_loader = DataLoader(patched_dataset, batch_size=1, shuffle=False)

with torch.no_grad():
    for i, test_input in enumerate(test_loader):
        test_input = test_input.to(device)
        reconstruction = model(test_input)

        # Calculate the absolute difference (Residual Map)
        residual_map = torch.abs(test_input - reconstruction)

        # Convert tensors back to numpy for visualization
        orig_hm = test_input.cpu().squeeze().numpy()
        recon_hm = reconstruction.cpu().squeeze().numpy()
        error_hm = residual_map.cpu().squeeze().numpy()

        # Post-Processing: Threshold the error to tightly bound the patch
        # Any error above 3x the average error is treated as the patch location
        threshold = error_hm.mean() + 1.5 * error_hm.std()
        localization_mask = (error_hm > threshold).astype(np.uint8) * 255

        # Plotting the results
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        axes[0].imshow(orig_hm, cmap='jet')
        axes[0].set_title("Input Patched Heatmap")

        axes[1].imshow(recon_hm, cmap='jet')
        axes[1].set_title("Autoencoder Reconstruction")

        axes[2].imshow(error_hm, cmap='hot')
        axes[2].set_title("Raw Residual Error")

        axes[3].imshow(localization_mask, cmap='gray')
        axes[3].set_title("Localized Patch Mask")

        plt.show()
        break  # Just checking the first instance for testing
