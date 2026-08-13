import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

from model import UNet
from dataset import BDD100kDataset

LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 10
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
TRAIN_IMG_DIR = "C:\\Research\\repos\\TestBed\\data\\train_images"
TRAIN_MASK_DIR = "C:\\Research\\repos\\TestBed\\data\\train_masks"
WEIGHTS_DIR = "C:\\Research\\repos\\TestBed\\encoders_to_patch_localization\\UNET\\weights"

def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)
    
    for batch_idx, batch in enumerate(loop):
        # Dataset may return (image, mask) or (image, mask, filename)
        if len(batch) == 3:
            data, targets, _ = batch
        else:
            data, targets = batch

        data = data.to(device=DEVICE)
        # Target shape must be (batch_size, H, W) and type Long for CrossEntropy
        targets = targets.long().to(device=DEVICE) 
        
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss=loss.item())

def main():
    # Albumentations is smart enough to know that A.Normalize should only be 
    # applied to the image, not the mask. Your class IDs (0–18, 255) inside the mask remain untouched.
    transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        # This converts the image and mask arrays into PyTorch Tensors.
        ToTensorV2(),
    ], is_check_shapes=False)

    model = UNet(in_channels=3, out_channels=19).to(DEVICE)
    
    # Crucial change: ignore_index=255
    loss_fn = nn.CrossEntropyLoss(ignore_index=255) 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_dataset = BDD100kDataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_MASK_DIR,
        transform=transform,
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Ensure weights directory exists
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        train_fn(train_loader, model, optimizer, loss_fn)
        torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, f"unet_bdd100k_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    main()