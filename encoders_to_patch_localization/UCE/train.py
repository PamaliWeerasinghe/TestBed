from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from prepare_and_preprocess_data import HeatmapDataset
from LightweightAutoencoder import LightweightAutoencoder

# Initialize hardware, dataset, loader, and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

project_root = Path(__file__).resolve().parents[2]
heatmap_dir = project_root / "heatmaps" / "clean"

if not heatmap_dir.exists():
    raise FileNotFoundError(f"Heatmap folder not found: {heatmap_dir}")

clean_dataset = HeatmapDataset(folder_path=str(heatmap_dir))
train_loader = DataLoader(clean_dataset, batch_size=16, shuffle=True)

model = LightweightAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 20  # Usually converges quickly for heatmaps
model.train()
for epoch in range(epochs):
    total_loss = 0
    for data in train_loader:
        inputs = data.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs) # Input is the target
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")
