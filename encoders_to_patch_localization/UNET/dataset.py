import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

class BDD100kDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        
        # BDD100k masks have identical filenames but end in _train_id.png
        mask_name = img_name.replace(".jpg", "_train_id.png")
        mask_path = os.path.join(self.mask_dir, mask_name)
        # print(f"Loading image: {img_path}, mask: {mask_path}")
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Load as grayscale (L), but keep values as integers (0-18, and 255)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.int64)
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask, img_name