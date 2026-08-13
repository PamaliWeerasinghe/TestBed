import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import UNet
from dataset import BDD100kDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
TEST_IMG_DIR = "C:\\Research\\repos\\TestBed\\data\\test_images"
TEST_MASK_DIR = "C:\\Research\\repos\\TestBed\\data\\test_masks"
OUTPUT_DIR = "C:\\Research\\repos\\TestBed\\data\\saved_predictions"

def save_predictions_as_imgs(loader, model, folder="C:\\Research\\repos\\TestBed\\data\\saved_predictions", device="cuda"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Standard BDD100k / Cityscapes RGB Color Palette
    color_map = np.array([
        [128, 64, 128],   # 0: Road (Purple)
        [244, 35, 232],   # 1: Sidewalk (Pink)
        [70, 70, 70],     # 2: Building (Dark Gray)
        [102, 102, 156],  # 3: Wall (Purple/Gray)
        [190, 153, 153],  # 4: Fence (Light Gray)
        [153, 153, 153],  # 5: Pole (Gray)
        [250, 170, 30],   # 6: Traffic Light (Orange)
        [220, 220, 0],    # 7: Traffic Sign (Yellow)
        [107, 142, 35],   # 8: Vegetation (Dark Green)
        [152, 251, 152],  # 9: Terrain (Light Green)
        [70, 130, 180],   # 10: Sky (Blue)
        [220, 20, 60],    # 11: Person (Red)
        [255, 0, 0],      # 12: Rider (Bright Red)
        [0, 0, 142],      # 13: Car (Dark Blue)
        [0, 0, 70],       # 14: Truck (Navy)
        [0, 60, 100],     # 15: Bus (Teal)
        [0, 80, 100],     # 16: Train (Dark Teal)
        [0, 0, 230],      # 17: Motorcycle (Blue)
        [119, 11, 32],    # 18: Bicycle (Dark Red)
    ], dtype=np.uint8)
        
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            # Get raw logits from the model
            logits = model(x)
            
            # Find the index (class ID) with the highest probability across the 19 channels
            preds = torch.argmax(logits, dim=1) 
            
        preds = preds.cpu().numpy()
        for i, pred_mask in enumerate(preds):
            # The saved image will look very dark because values only go from 0 to 18
            img = Image.fromarray(pred_mask.astype(np.uint8))
            img.save(f"{folder}/pred_{idx}_{i}.png")
            
    model.train()
    print(f"Predictions saved to {folder}")

def main():
    transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ], is_check_shapes=False)

    model = UNet(in_channels=3, out_channels=19).to(DEVICE)
    model.load_state_dict(torch.load("unet_bdd100k_epoch_10.pth")) # Load correct weights
    
    test_dataset = BDD100kDataset(image_dir=TEST_IMG_DIR, mask_dir=TEST_MASK_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    save_predictions_as_imgs(test_loader, model, folder=OUTPUT_DIR, device=DEVICE)

if __name__ == "__main__":
    main()