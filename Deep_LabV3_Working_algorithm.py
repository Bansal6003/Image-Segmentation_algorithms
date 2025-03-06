import os
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import mlflow
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image as grayscale
        image = Image.open(self.image_paths[idx]).convert('L')
        mask = np.array(Image.open(self.mask_paths[idx]))
        
        # Convert grayscale to 3 channels
        if self.transform:
            image = self.transform(image)
            # Duplicate grayscale channel to 3 channels
            image = image.repeat(3, 1, 1)
        
        # Process mask: convert to binary
        processed_mask = (mask > 0).astype(np.float32)
            
        # Resize mask
        mask_pil = Image.fromarray(processed_mask)
        mask_resized = transforms.Resize((512, 512), 
                                       interpolation=transforms.InterpolationMode.NEAREST)(mask_pil)
        mask_tensor = torch.from_numpy(np.array(mask_resized))
        
        return image, mask_tensor.float()  # Use float for binary segmentation

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=500):
    model.to(device)
    best_val_loss = float('inf')
    patience = 200  
    patience_counter = 0
    scaler = torch.cuda.amp.GradScaler()
    
    # Lists to store metrics
    train_losses = []
    val_losses = []
    
    try:
        with mlflow.start_run():
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("initial_lr", optimizer.param_groups[0]['lr'])
            
            for epoch in range(num_epochs):
                # Training phase
                model.train()
                train_loss = 0
                batch_count = 0
                
                for images, masks in train_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    optimizer.zero_grad()
                    
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, masks.unsqueeze(1))  # Add channel dimension
                    
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.item()
                    batch_count += 1
                
                avg_train_loss = train_loss / batch_count
                train_losses.append(avg_train_loss)
                
                # Validation phase
                model.eval()
                val_loss = 0
                batch_count = 0
                
                with torch.no_grad():
                    for images, masks in val_loader:
                        images = images.to(device)
                        masks = masks.to(device)
                        
                        with torch.cuda.amp.autocast():
                            outputs = model(images)
                            loss = criterion(outputs, masks.unsqueeze(1))
                        
                        val_loss += loss.item()
                        batch_count += 1
                
                avg_val_loss = val_loss / batch_count
                val_losses.append(avg_val_loss)
                
                # Update learning rate
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                
                # Print progress
                print(f'Epoch [{epoch+1}/{num_epochs}]')
                print(f'Training Loss: {avg_train_loss:.4f}')
                print(f'Validation Loss: {avg_val_loss:.4f}')
                print(f'Learning Rate: {current_lr:.6f}')
                
                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': best_val_loss,
                    }, 'best_model_checkpoint.pth')
                    print("Saved new best model")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered")
                        break
                
                # Log metrics
                mlflow.log_metrics({
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "learning_rate": current_lr
                }, step=epoch)
                
                # Plot losses every 10 epochs
                if (epoch + 1) % 10 == 0:
                    plt.figure(figsize=(10, 5))
                    plt.plot(train_losses, label='Training Loss')
                    plt.plot(val_losses, label='Validation Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.savefig('loss_plot.png')
                    plt.close()
                    mlflow.log_artifact('loss_plot.png')
    
    except Exception as e:
        print(f"Training error: {str(e)}")
        raise e
    finally:
        mlflow.end_run()
    
    return model

def main():
    # Dataset Paths
    image_dir = r'C:\Users\Pushkar Bansal\Downloads\wt_2.5x_3dpf_yolkext.v4i.yolov11\train\images'
    mask_dir = r'C:\Users\Pushkar Bansal\Downloads\wt_2.5x_3dpf_yolkext.v4i.yolov11\masks'
    
    # Debug directory existence
    print("\nChecking directories:")
    print(f"Image directory exists: {os.path.exists(image_dir)}")
    print(f"Mask directory exists: {os.path.exists(mask_dir)}")
    
    # Get all files
    image_files = os.listdir(image_dir)
    mask_files = os.listdir(mask_dir)
    
    print("\nFirst few files found:")
    print("Images:", image_files[:5])
    print("Masks:", mask_files[:5])
    
    # Get file paths with supported extensions
    image_paths = sorted([os.path.join(image_dir, f) for f in image_files 
                         if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in mask_files 
                        if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff'))])
    
    print(f"\nFound {len(image_paths)} images and {len(mask_paths)} masks")
    
    # Create a dict of filenames without extensions for matching
    image_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in image_paths}
    
    # For masks, remove the "_mask_0" suffix before using as key
    mask_dict = {os.path.splitext(os.path.basename(p))[0].replace('_mask_0', ''): p for p in mask_paths}
    
    print("\nExample filenames after processing:")
    print("Image names:", list(image_dict.keys())[:5])
    print("Processed mask names:", list(mask_dict.keys())[:5])
    
    # Find common filenames
    common_names = set(image_dict.keys()) & set(mask_dict.keys())
    print(f"\nFound {len(common_names)} matching image-mask pairs")
    if common_names:
        print("Example matching names:", list(common_names)[:5])
    
    # Create lists of matched pairs
    matched_image_paths = [image_dict[name] for name in common_names]
    matched_mask_paths = [mask_dict[name] for name in common_names]
    
    # Verify all files exist
    valid_pairs = []
    for img_path, mask_path in zip(matched_image_paths, matched_mask_paths):
        if os.path.exists(img_path) and os.path.exists(mask_path):
            valid_pairs.append((img_path, mask_path))
        else:
            print(f"\nMissing file:")
            print(f"Image exists: {os.path.exists(img_path)} - {img_path}")
            print(f"Mask exists: {os.path.exists(mask_path)} - {mask_path}")
    
    print(f"\nFinal number of valid pairs: {len(valid_pairs)}")
    
    if not valid_pairs:
        print("\nNo valid pairs found. This could be due to:")
        print("1. Different naming conventions between images and masks")
        print("2. Different file extensions")
        print("3. Missing corresponding files")
        print("4. Case sensitivity in filenames")
        raise ValueError("No valid image-mask pairs found! Check the debug information above.")
    
    # Print first few pairs to verify matching
    print("\nFirst few matched pairs:")
    for img, mask in valid_pairs[:3]:
        print(f"Image: {os.path.basename(img)}")
        print(f"Mask: {os.path.basename(mask)}")
    
    # Unzip the valid pairs
    image_paths, mask_paths = zip(*valid_pairs)
    
    # Data transforms for grayscale images
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    
    # Split dataset
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    

    # Create datasets
    train_dataset = SegmentationDataset(train_images, train_masks, transform)
    val_dataset = SegmentationDataset(val_images, val_masks, transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=8,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )
    
    # Initialize model for binary segmentation
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,  # Binary segmentation
    )
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Better for binary segmentation
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Cosine annealing scheduler with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Initial period
        T_mult=2,  # Period multiplier
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Train model
    print("\nStarting training...")
    try:
        trained_model = train_model(
            model, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer,
            scheduler,
            num_epochs=100  # Reduced epochs with better scheduling
        )
        
        # Save final model
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'num_classes': 1,
        }, 'deeplabv3plus_final_model.pth')
        print("\nTraining completed and model saved")
        
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        raise e

if __name__ == "__main__":
    main()