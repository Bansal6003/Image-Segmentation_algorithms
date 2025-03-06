import sys
import os
import torch
import torchvision.transforms as transforms
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget, QMessageBox, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_class):
        super(UNet, self).__init__()
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


class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Mask Area Prediction')
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Create and add imageLabel
        self.imageLabel = QLabel(self)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setMinimumSize(400, 400)  # Set a minimum size for the image display
        layout.addWidget(self.imageLabel)

        self.uploadButton = QPushButton('Upload Folder and Process', self)
        self.uploadButton.clicked.connect(self.upload_folder)
        layout.addWidget(self.uploadButton)

        self.backButton = QPushButton('Back', self)
        self.backButton.clicked.connect(self.show_previous_image)
        self.backButton.setEnabled(False)
        layout.addWidget(self.backButton)

        self.forwardButton = QPushButton('Forward', self)
        self.forwardButton.clicked.connect(self.show_next_image)
        self.forwardButton.setEnabled(False)
        layout.addWidget(self.forwardButton)

        self.images = []
        self.image_paths = []
        self.processed_images = []
        self.current_index = -1
        self.models = {}

    def load_models(self):
        # Define model paths
        model_paths = {
            'eye': r'D:\Behavioral genetics_V1\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Collective_Unet_models\UNet_model_eye.pth'
        }
    
        for class_name, model_path in model_paths.items():
            model = UNet(n_class=1)
            model.load_state_dict(torch.load(model_path))
            model = model.to('cuda')
            self.models[class_name] = model

    def upload_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder Containing Test Images')
        if not folder_path:
            return

        self.load_models()

        pixel_size = 0.008  # mm

        self.images = []
        self.image_paths = []
        self.processed_images = []
        self.current_index = -1

        # Create an output directory
        output_directory = os.path.join(folder_path, "processed_images")
        os.makedirs(output_directory, exist_ok=True)

        # Initialize DataFrame with size columns for each class
        self.df = pd.DataFrame(columns=["Image"] + [f'{class_name}_Mask Size (mm²)' for class_name in self.models.keys()])

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                image_path = os.path.join(folder_path, filename)
                original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
                if original_image is not None:
                    self.images.append(original_image)
                    self.image_paths.append(image_path)

                    self.process_all_classes(image_path, pixel_size, output_directory)

        # Save results to Excel file
        output_file = os.path.join(folder_path, "mask_sizes_mm2.xlsx")
        self.df.to_excel(output_file, index=False)

        if self.images:
            self.current_index = 0
            self.display_image(self.images[self.current_index])
            self.update_buttons()

            QMessageBox.information(self, 'Success', f'Mask sizes saved to {output_file}.')
        else:
            QMessageBox.warning(self, 'Warning', 'No images processed successfully.')

    def process_all_classes(self, image_path, pixel_size, output_directory):
        # Dictionary to store mask sizes with the image name
        mask_sizes = {"Image": os.path.basename(image_path)}
        
        for class_name, model in self.models.items():
            # Process the image to get the mask area and the mask
            mask_area_mm2, mask = self.process_image(image_path, model)

            # Store the mask area in mm²
            mask_sizes[f'{class_name}_Mask Size (mm²)'] = mask_area_mm2

            # Save the mask image
            mask_output_path = os.path.join(output_directory, f'{class_name}_mask_{os.path.basename(image_path)}')
            cv2.imwrite(mask_output_path, mask)

        # Add the mask sizes for this image to the DataFrame using concat
        new_row = pd.DataFrame([mask_sizes])
        self.df = pd.concat([self.df, new_row], ignore_index=True)

    def process_image(self, image_path, model):
        # Load and preprocess the grayscale image
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        original_image = cv2.resize(original_image, (256, 256))  # Resize as required for the model
        original_image_tensor = torch.tensor(original_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to('cuda') / 255.0

        # Make predictions
        with torch.no_grad():
            model.eval()
            output = model(original_image_tensor)
            predicted_mask = output.squeeze(0).squeeze(0).cpu().numpy()

        # Convert mask predictions to binary
        predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8)

        # Calculate mask area in mm²
        pixel_size = 0.008
        pixel_area_mm2 = (pixel_size ** 2)  # mm²
        mask_area_mm2 = np.sum(predicted_mask_binary) * pixel_area_mm2

        # Prepare mask for saving (ensure it's 0-255 range)
        mask_to_save = (predicted_mask_binary * 255).astype(np.uint8)

        return mask_area_mm2, mask_to_save

    def display_image(self, image):
        if isinstance(image, np.ndarray):
            # Convert OpenCV image (numpy array) to QImage
            height, width = image.shape[:2]
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        elif isinstance(image, str):
            # If image is a file path, load it directly
            q_image = QImage(image)
        else:
            print("Unsupported image type")
            return

        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.imageLabel.setPixmap(scaled_pixmap)

    def show_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_image(self.images[self.current_index])
            self.update_buttons()

    def show_next_image(self):
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.display_image(self.images[self.current_index])
            self.update_buttons()

    def update_buttons(self):
        self.backButton.setEnabled(self.current_index > 0)
        self.forwardButton.setEnabled(self.current_index < len(self.images) - 1)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessor()
    ex.show()
    sys.exit(app.exec_())
