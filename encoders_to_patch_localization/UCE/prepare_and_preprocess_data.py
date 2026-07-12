import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


class HeatmapDataset(Dataset):
    def __init__(self, folder_path, img_size=(128, 128)):
        self.folder_path = folder_path
        self.img_paths = []
        self.img_size = img_size

        root = Path(folder_path)
        subfolders = [p for p in root.iterdir() if p.is_dir()]

        if subfolders:
            for subfolder in subfolders:
                image_name = f"{subfolder.name}_cam_rgb.jpg"
                image_path = subfolder / image_name
                if image_path.exists():
                    self.img_paths.append(str(image_path))
        else:
            for file_path in root.iterdir():
                if file_path.is_file():
                    self.img_paths.append(str(file_path))

        if not self.img_paths:
            raise FileNotFoundError(
                f"No heatmap images found under {folder_path}. Expected subfolders containing '<name>_cam_rgb.jpg'."
            )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        # EigenCAM heatmaps are usually saved as grayscale or RGB. Read as grayscale.
        heatmap = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        heatmap = cv2.resize(heatmap, self.img_size)

        # Normalize to [0, 1] and add channel dimension
        heatmap = heatmap.astype(np.float32) / 255.0
        heatmap = np.expand_dims(heatmap, axis=0)

        return torch.tensor(heatmap)
