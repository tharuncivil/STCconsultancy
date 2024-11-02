import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import rasterio
from sklearn.metrics import accuracy_score, f1_score

# Define the U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        
        self.down1 = double_conv(in_channels, 64)
        self.down2 = double_conv(64, 128)
        self.down3 = double_conv(128, 256)
        self.down4 = double_conv(256, 512)
        
        self.up1 = double_conv(512 + 256, 256)
        self.up2 = double_conv(256 + 128, 128)
        self.up3 = double_conv(128 + 64, 64)
        self.up4 = double_conv(64, 64)
        
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        c1 = self.down1(x)
        x = self.pool(c1)
        c2 = self.down2(x)
        x = self.pool(c2)
        c3 = self.down3(x)
        x = self.pool(c3)
        c4 = self.down4(x)
        
        x = self.upsample(c4)
        x = torch.cat([x, c3], dim=1)
        x = self.up1(x)
        x = self.upsample(x)
        x = torch.cat([x, c2], dim=1)
        x = self.up2(x)
        x = self.upsample(x)
        x = torch.cat([x, c1], dim=1)
        x = self.up3(x)
        x = self.up4(x)
        
        return self.out(x)

# Custom Dataset
class LULCDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        with rasterio.open(self.image_paths[idx]) as src:
            image = src.read().transpose(1, 2, 0)  # CxHxW to HxWxC
        
        with rasterio.open(self.mask_paths[idx]) as src:
            mask = src.read(1)
        
        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(mask).long()
        
        return image, mask

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

# Evaluation function
def evaluate(model, val_loader, device):
    model.eval()
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_masks.extend(masks.numpy().flatten())
    
    accuracy = accuracy_score(all_masks, all_preds)
    f1 = f1_score(all_masks, all_preds, average='weighted')
    
    return accuracy, f1

# Main execution
def main():
    # Set up parameters
    in_channels = 4  # Assuming 4-band imagery (RGB + NIR)
    out_channels = 10  # Number of LULC classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = UNet(in_channels, out_channels).to(device)
    
    # Set up data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.25])
    ])
    
    # Assume you have lists of image and mask file paths
    train_dataset = LULCDataset(train_image_paths, train_mask_paths, transform)
    val_dataset = LULCDataset(val_image_paths, val_mask_paths, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    # Set up training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, device)
        accuracy, f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), 'lulc_model.pth')

if __name__ == '__main__':
    main()