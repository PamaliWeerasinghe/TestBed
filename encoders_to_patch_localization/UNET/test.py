import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import UNet
from dataset import BDD100kDataset

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
TEST_IMG_DIR = "C:\\Research\\repos\\TestBed\\data\\test_images"
TEST_MASK_DIR = "C:\\Research\\repos\\TestBed\\data\\test_masks"
OUTPUT_DIR = "C:\\Research\\repos\\TestBed\\data\\saved_predictions"
WEIGHTS_DIR = "C:\\Research\\repos\\TestBed\\encoders_to_patch_localization\\UNET\\weights"

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
    for idx, batch in enumerate(loader):
        # Dataset may return (image, mask) or (image, mask, filename)
        if len(batch) == 3:
            x, y, names = batch
        else:
            x, y = batch
            names = [f"{idx}_{i}.jpg" for i in range(x.shape[0])]

        x = x.to(device=device)
        with torch.no_grad():
            logits = model(x)
            preds = torch.argmax(logits, dim=1) 

        preds = preds.cpu().numpy()

        for i, pred_mask in enumerate(preds):
            # Map the integer class IDs to the RGB colors
            # Any class outside 0-18 (like 255) will cause an error,
            # so we clip it to a safe index just for visualization
            safe_mask = np.clip(pred_mask, 0, 18)
            rgb_mask = color_map[safe_mask]

            img = Image.fromarray(rgb_mask)
            # Use original filename when available, otherwise fall back to index-based name
            orig_name = names[i]
            base, _ = os.path.splitext(orig_name)
            out_name = f"pred_{base}.png"
            img.save(os.path.join(folder, out_name))
            
    model.train()
    print(f"Predictions saved to {folder}")

def main():
    transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ], is_check_shapes=False)

    model = UNet(in_channels=3, out_channels=19).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_DIR + "\\unet_bdd100k_epoch_10.pth")) # Load correct weights
    
    test_dataset = BDD100kDataset(image_dir=TEST_IMG_DIR, mask_dir=TEST_MASK_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    save_predictions_as_imgs(test_loader, model, folder=OUTPUT_DIR, device=DEVICE)

if __name__ == "__main__":
    main()