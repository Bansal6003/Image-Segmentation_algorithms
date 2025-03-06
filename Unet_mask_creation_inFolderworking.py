import sys
import os
import torch
import torchvision.transforms as transforms
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QComboBox,QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

class UNet(nn.Module):
    def __init__(self, n_class):
        super(UNet, self).__init__()
        # Define the layers for the U-Net as in your original implementation
        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
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

        # Decoder layers
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


class SegmentationApp(QWidget):
    def __init__(self, model_paths):
        super().__init__()
        self.model_paths = model_paths
        self.models = {}
        self.current_image_index = 0
        self.images = []
        self.load_models()
        
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        # ComboBox for selecting models
        self.model_selector = QComboBox(self)
        self.model_selector.addItems(list(self.models.keys()))
        layout.addWidget(self.model_selector)
        
        # Button to upload folder
        self.upload_button = QPushButton('Upload Folder', self)
        self.upload_button.clicked.connect(self.upload_folder)
        layout.addWidget(self.upload_button)
        
        # QLabel for displaying images
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)
        
        # QLabel for displaying masks
        self.mask_label = QLabel(self)
        layout.addWidget(self.mask_label)
        
        # Buttons to navigate through images
        self.next_button = QPushButton('Next Image', self)
        self.next_button.clicked.connect(self.show_next_image)
        layout.addWidget(self.next_button)
        
        self.prev_button = QPushButton('Previous Image', self)
        self.prev_button.clicked.connect(self.show_prev_image)
        layout.addWidget(self.prev_button)
        
        # Set the layout
        self.setLayout(layout)

    def upload_folder(self):
        # Open a file dialog to select a folder
        folder = QFileDialog.getExistingDirectory(self, 'Select Folder')
        if folder:
            self.images = self.load_images(folder)
            if self.images:
                self.current_image_index = 0
                self.show_image_and_mask(self.current_image_index)

    def load_images(self, folder):
        # Load all grayscale images from the folder
        return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg') or f.endswith('.png')]

    def load_models(self):
        # Load all models and move to GPU if available
        for model_name, model_path in self.model_paths.items():
            model = UNet(n_class=1)
            model.load_state_dict(torch.load(model_path))
            model.eval()  # Set the model to evaluation mode
            model.to('cuda')  # Move model to GPU
            self.models[model_name] = model

    def pil_to_qt(self, pil_image):
        pil_image = pil_image.convert('L')  # Ensure grayscale format
        image_data = pil_image.tobytes("raw", "L")
        width, height = pil_image.size
        return QImage(image_data, width, height, QImage.Format_Grayscale8)

    def preprocess_image(self, image_path, transform):
        image = Image.open(image_path).convert('L')  # Load as grayscale
        if transform:
            image = transform(image)
        return image.unsqueeze(0)  # Add batch dimension

    def postprocess_output(self, output):
        output = torch.sigmoid(output)  # Apply sigmoid for binary output
        output = (output > 0.5).float()  # Binarize output
        mask = output.squeeze(0).squeeze(0)  # Remove batch and channel dimension
        return mask.detach().cpu().numpy()  # Convert to numpy

    def show_image_and_mask(self, index):
        if self.images:
            image_path = self.images[index]
            original_image = Image.open(image_path).convert('L')  # Load original for display
            
            # Get the selected model
            selected_model_name = self.model_selector.currentText()
            selected_model = self.models[selected_model_name]
            
            # Preprocess the image
            transform = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize to smaller size (128, 128)
                transforms.ToTensor()
            ])
            input_image = self.preprocess_image(image_path, transform).to('cuda')  # Move input to GPU
            
            # Get model prediction
            with torch.no_grad():
                output = selected_model(input_image)
            
            # Postprocess output to get segmentation mask
            segmentation_mask = self.postprocess_output(output)
            
            # Calculate mask area
            mask_area = np.count_nonzero(segmentation_mask)  # Number of non-zero pixels
            print(f"Mask Area: {mask_area} pixels")
            
            # Resize the original image for display (smaller size, e.g., 128x128)
            resized_image = original_image.resize((256, 256))
            
            # Convert images to QImage and display
            qt_image = self.pil_to_qt(resized_image)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))
            
            # Convert mask to QImage and resize it for display
            mask_image = Image.fromarray((segmentation_mask * 255).astype('uint8')).resize((256, 256))
            qt_mask = self.pil_to_qt(mask_image)
            self.mask_label.setPixmap(QPixmap.fromImage(qt_mask))
    
            # Resize the window to accommodate the smaller images
            self.resize(300, 300)  # Resize the window (adjust as needed)

    def show_next_image(self):
        if self.current_image_index < len(self.images) - 1:
            self.current_image_index += 1
            self.show_image_and_mask(self.current_image_index)

    def show_prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image_and_mask(self.current_image_index)

def main():
    model_paths = {
        'Model 1': r"D:\Behavioral genetics_V1\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Collective_Unet_models\UNet_model_eye.pth",
        # 'Model 2': r'D:\Behavioral genetics_V1\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Collective_Unet_models\UNet_model_yolkext_tail.pth'
        # Add more models as needed
    }

    app = QApplication(sys.argv)
    window = SegmentationApp(model_paths)
    window.setWindowTitle('Image Segmentation Viewer')
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()