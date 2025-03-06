import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt

# Custom Dataset class to handle grayscale images and polygon labels
class PolygonDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, os.path.splitext(self.image_filenames[idx])[0] + '.png')  # Updated to .png

        # Load grayscale image
        try:
            image = Image.open(img_path).convert('L')  # 'L' mode for grayscale
        except FileNotFoundError:
            print(f"Warning: Image file not found: {img_path}")
            # Return a blank image or handle it in a way that suits your application
            image = Image.new('L', (224, 224))  # Default image of size 224x224

        # Load mask image
        try:
            mask = Image.open(mask_path).convert('L')  # Assuming mask images are in grayscale format
        except FileNotFoundError:
            print(f"Warning: Mask file not found: {mask_path}")
            # Return a blank mask or handle it as needed
            mask = Image.new('L', (224, 224))  # Default mask of size 224x224

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask  # Return image and mask

# U-Net model modified for grayscale images
class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        # Encoder
        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # Changed input channels to 1
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        xe11 = torch.relu(self.e11(x))
        xe12 = torch.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = torch.relu(self.e21(xp1))
        xe22 = torch.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = torch.relu(self.e31(xp2))
        xe32 = torch.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = torch.relu(self.e41(xp3))
        xe42 = torch.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = torch.relu(self.e51(xp4))
        xe52 = torch.relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = torch.relu(self.d11(xu11))
        xd12 = torch.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = torch.relu(self.d21(xu22))
        xd22 = torch.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = torch.relu(self.d31(xu33))
        xd32 = torch.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = torch.relu(self.d41(xu44))
        xd42 = torch.relu(self.d42(xd41))

        out = self.outconv(xd42)
        return out

# Main function
def main():
    # Data directories
    train_image_dir = r"C:\Users\Pushkar Bansal\Downloads\48_dpf_new_params_eye.v7i.yolov9\train\images"
    train_mask_dir = r"C:\Users\Pushkar Bansal\Downloads\48_dpf_new_params_eye.v7i.yolov9\train\masks"
    
    # Data transformation
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    # Create dataset and dataloader
    train_dataset = PolygonDataset(image_dir=train_image_dir, mask_dir=train_mask_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Model, loss, and optimizer
    net = UNet(n_class=1).to('cuda')  # Assuming binary segmentation
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy for binary segmentation
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)
    
    # Initialize variables for saving the best model
    best_loss = float('inf')  # Track best loss
    model_path = r'D:\Behavioral genetics_V1\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Collective_Unet_models\UNet_model_eye.pth'  # Path to save the model
    
    # Training loop
    EPOCHS = 500
    average_losses = []
    running_losses = []
    
    for epoch in range(EPOCHS):
        losses = []
        running_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
    
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
    
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
    
            if i % 1 == 0:
                print(f'Epoch: {epoch + 1}, minibatch: {i}, loss: {loss.item():.4f}')
    
        avg_loss = sum(losses) / len(losses)
        average_losses.append(avg_loss)
        running_losses.append(running_loss)
    
        # Save the best model based on validation loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(net.state_dict(), model_path)
            print(f'Best model saved with loss: {best_loss:.4f}')
    
        scheduler.step(avg_loss)
    
    # Plot the loss
    plt.plot(average_losses)
    plt.title('Average Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    main()
