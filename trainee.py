from data_gen import generate_random_shapes_image, blur_image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import torch.amp
import seaborn as sns

from matplotlib import pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])]
)
sigma_range = (0, 30)
sigma_mean = (sigma_range[1] - sigma_range[0]) / 2
sigma_std = (sigma_range[1] - sigma_range[0]) / 2
import os
from pathlib import Path
import h5py


def precompute_dataset(length=12800, save_dir="blur_dataset"):
    Path(save_dir).mkdir(exist_ok=True)
    with h5py.File(f"{save_dir}/dataset.h5", "w") as f:
        images = f.create_dataset("images", (length, 1, 128, 128), dtype=np.float32)
        sigmas = f.create_dataset("sigmas", (length,), dtype=np.float32)

        for i in tqdm(range(length), desc="Precomputing dataset"):
            image = generate_random_shapes_image()
            image_np = np.asarray(image)
            sigma = np.random.uniform(sigma_range[0], sigma_range[1])
            blurred = blur_image(image_np, sigma)
            blurred_pil = Image.fromarray(blurred)
            blurred_tensor = transform(blurred_pil)
            images[i] = blurred_tensor.numpy()
            sigmas[i] = (sigma - sigma_mean) / sigma_std

    return save_dir


class PrecomputedBlurDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_file = h5py.File(h5_path, "r")
        self.images = self.h5_file["images"]
        self.sigmas = self.h5_file["sigmas"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx])
        sigma = torch.tensor(self.sigmas[idx], dtype=torch.float32)
        return image, sigma

    def __del__(self):
        self.h5_file.close()


# In main():

# Custom Dataset
# class BlurDataset(Dataset):
#     def __init__(self, length=10000):
#         self.length = length
#
#     def __len__(self):
#         return self.length
#
#     def __getitem__(self, idx):
#         image = generate_random_shapes_image()
#
#         # Convert to numpy for OpenCV blur
#         image_np = np.asarray(image)
#
#         # Generate random sigma and apply Gaussian blur
#         sigma = np.random.uniform(sigma_range[0], sigma_range[1])
#         blurred = blur_image(image_np, sigma)
#         blurred_pil = Image.fromarray(blurred)  # Convert back to PIL for transforms
#
#         blurred_tensor = transform(blurred_pil)
#
#         if blurred_tensor.shape != (1, 128, 128):
#             raise ValueError(f"Unexpected tensor shape: {blurred_tensor.shape}")
#
#         # Normalize sigma to be between 0 and 1
#
#         sigma_normalized = (sigma - sigma_mean) / sigma_std
#         return blurred_tensor, torch.tensor(sigma_normalized, dtype=torch.float32)


# CNN Model
class BlurRegressionCNN(nn.Module):
    def __init__(self):
        super(BlurRegressionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(-1)


# Training function
def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")
    model_path = "best_blur_model.pth"

    losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, sigmas in tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        ):
            images, sigmas = images.to(device), sigmas.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, sigmas)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, sigmas in val_loader:
                images, sigmas = images.to(device), sigmas.to(device)
                outputs = model(images)
                loss = criterion(outputs, sigmas)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        losses.append((train_loss, val_loss))
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model with Val Loss: {val_loss:.4f}")

        scheduler.step()
    plt.plot([l[0] for l in losses], label="Training Loss")
    plt.plot([l[1] for l in losses], label="Validation Loss")
    plt.legend()
    plt.yscale("log")
    plt.show()

    return model


def main():
    # Hyperparameters
    batch_size = 128
    num_epochs = 200
    learning_rate = 1e-3
    val_split = 0.2

    # Dataset
    # dataset = BlurDataset(length=12800)
    save_dir = precompute_dataset(length=12800)
    dataset = PrecomputedBlurDataset(f"{save_dir}/dataset.h5")


    # Split into train and validation
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    global model
    # Model
    model = BlurRegressionCNN().to(device)

    # Train
    model = train_model(model, train_loader, val_loader, num_epochs, learning_rate)

    print("Training completed!")

if __name__ == "__main__":
    main()
