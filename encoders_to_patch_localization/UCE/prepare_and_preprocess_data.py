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
    def __init__(self, folder_path, img_size=None):
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
                if file_path.is_file() and file_path.name.endswith("_cam_rgb.jpg"):
                    self.img_paths.append(str(file_path))

        if not self.img_paths:
            raise FileNotFoundError(
                f"No RGB heatmap images found under {folder_path}. Expected files ending with '_cam_rgb.jpg'."
            )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        heatmap = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if heatmap is None:
            raise ValueError(f"Failed to read image: {img_path}")

        if self.img_size is not None:
            heatmap = cv2.resize(heatmap, self.img_size)

        heatmap = heatmap.astype(np.float32) / 255.0
        heatmap = np.expand_dims(heatmap, axis=0)

        return torch.tensor(heatmap)
