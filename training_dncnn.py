import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio
import matplotlib.pyplot as plt

# Define the DnCNN model with Xavier initialization
class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=3, kernel_size=3):
        super(DnCNN, self).__init__()
        layers = []
        # First layer: Conv + ReLU
        layers.append(nn.Conv2d(image_channels, n_channels, kernel_size, padding=kernel_size//2))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers: Conv + BatchNorm + ReLU
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.BatchNorm2d(n_channels))
            layers.append(nn.ReLU(inplace=True))

        # Last layer: Conv
        layers.append(nn.Conv2d(n_channels, image_channels, kernel_size, padding=kernel_size//2))

        self.dncnn = nn.Sequential(*layers)

        # Apply Xavier initialization to all convolutional layers
        for m in self.dncnn.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.dncnn(x)

# Custom Dataset
class DenoisingDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        noisy_path = self.data.iloc[idx]['noisy_path']
        clean_path = self.data.iloc[idx]['clean_path']

        try:
            noisy_img = Image.open(noisy_path).convert('RGB')
            clean_img = Image.open(clean_path).convert('RGB')
        except (FileNotFoundError, IOError, ValueError) as e:
            print(f"Error loading image at index {idx}: {e}")
            # Return zero tensors as a fallback
            noisy_img = Image.new('RGB', (256, 256))  # Default size
            clean_img = Image.new('RGB', (256, 256))

        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)

        return noisy_img, clean_img

# Validation function
def validate_model(model, val_loader, criterion, device, epoch, save_dir='val_images'):
    model.eval()
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    total_psnr = 0.0
    total_loss = 0.0
    samples_to_save = 3  # Number of sample images to save
    saved = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for noisy_imgs, clean_imgs in val_loader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            outputs = model(noisy_imgs)
            denoised_imgs = noisy_imgs - outputs

            # Compute loss
            loss = criterion(outputs, clean_imgs - noisy_imgs)
            total_loss += loss.item() * noisy_imgs.size(0)

            # Compute PSNR
            denoised_imgs = torch.clamp(denoised_imgs, 0.0, 1.0)
            psnr = psnr_metric(denoised_imgs, clean_imgs)
            total_psnr += psnr.item() * noisy_imgs.size(0)

            # Save sample images
            if saved < samples_to_save:
                for i in range(min(noisy_imgs.size(0), samples_to_save - saved)):
                    noisy = noisy_imgs[i].cpu().permute(1, 2, 0).numpy()
                    clean = clean_imgs[i].cpu().permute(1, 2, 0).numpy()
                    denoised = denoised_imgs[i].cpu().permute(1, 2, 0).numpy()

                    plt.figure(figsize=(15, 5))
                    plt.subplot(1, 3, 1)
                    plt.imshow(noisy)
                    plt.title('Noisy')
                    plt.axis('off')
                    plt.subplot(1, 3, 2)
                    plt.imshow(denoised)
                    plt.title('Denoised')
                    plt.axis('off')
                    plt.subplot(1, 3, 3)
                    plt.imshow(clean)
                    plt.title('Clean')
                    plt.axis('off')
                    plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_sample_{saved+i}.png'))
                    plt.close()
                saved += noisy_imgs.size(0)

    avg_psnr = total_psnr / len(val_loader.dataset)
    avg_loss = total_loss / len(val_loader.dataset)
    print(f'Validation - Epoch {epoch}, Loss: {avg_loss:.6f}, PSNR: {avg_psnr:.2f} dB')
    model.train()
    return avg_psnr

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for noisy_imgs, clean_imgs in train_loader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs - noisy_imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * noisy_imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Training - Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}')

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0:
            validate_model(model, val_loader, criterion, device, epoch + 1)

# Main script
def main():
    # Hyperparameters
    batch_size = 16
    num_epochs = 30
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transforms with resizing for consistency
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Ensure consistent size
        transforms.ToTensor()
    ])

    # Load dataset
    dataset = DenoisingDataset(csv_file=r'D:\CV_project\dncnn training\traning_dncnn.csv', transform=transform)
    
    # Split into train and validation (80-20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Use single worker to debug
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model with Xavier initialization
    model = DnCNN(image_channels=3).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    # Save the model
    torch.save(model.state_dict(), 'dncnn_color.pth')
    print("Model saved as 'dncnn_color.pth'")

if __name__ == '__main__':
    main()