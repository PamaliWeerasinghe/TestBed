import os
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
        self.img_names = os.listdir(folder_path)
        self.img_size = img_size

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.img_names[idx])
        # EigenCAM heatmaps are usually saved as grayscale or RGB. Read as grayscale.
        heatmap = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        heatmap = cv2.resize(heatmap, self.img_size)
        
        # Normalize to [0, 1] and add channel dimension
        heatmap = heatmap.astype(np.float32) / 255.0
        heatmap = np.expand_dims(heatmap, axis=0) 
        
        return torch.tensor(heatmap)
