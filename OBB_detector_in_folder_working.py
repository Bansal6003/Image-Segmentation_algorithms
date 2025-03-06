from ultralytics import YOLO
import cv2
import os
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget, QMessageBox, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('YOLOv8 Detection')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.label = QLabel(self)
        layout.addWidget(self.label)

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

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.images = []
        self.image_paths = []
        self.processed_images = []
        self.current_index = -1

        # Load the YOLO model
        self.model = YOLO(r'D:\Behavioral genetics_V1\AI_Project_Python_Templates\Annotation for yolo and coco\runs\obb\train2\weights\best.pt')

    def upload_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder Containing Test Images')
        if not folder_path:
            return

        self.images = []
        self.image_paths = []
        self.processed_images = []
        self.current_index = -1

        output_directory = os.path.join(folder_path, "processed_images")
        os.makedirs(output_directory, exist_ok=True)

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                image_path = os.path.join(folder_path, filename)
                original_image = cv2.imread(image_path)
                if original_image is not None:
                    self.image_paths.append(image_path)

                    # Process the image
                    results = self.model(image_path, show=False, save=True)
                    predicted_img = results[0].plot()

                    # Resize the image for display
                    resized_img = self.resize_image(predicted_img, 800, 600)
                    self.processed_images.append(resized_img)

                    # Save the processed image
                    output_image_path = os.path.join(output_directory, f'processed_{filename}')
                    cv2.imwrite(output_image_path, predicted_img)

        if self.processed_images:
            self.current_index = 0
            self.display_image(self.processed_images[self.current_index])
            self.update_navigation_buttons()
            QMessageBox.information(self, 'Success', 'Images processed and saved.')
        else:
            QMessageBox.warning(self, 'Warning', 'No images were processed.')

    def resize_image(self, image, new_width, new_height):
        return cv2.resize(image, (new_width, new_height))

    def display_image(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(q_image))

    def show_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_image(self.processed_images[self.current_index])
        self.update_navigation_buttons()

    def show_next_image(self):
        if self.current_index < len(self.processed_images) - 1:
            self.current_index += 1
            self.display_image(self.processed_images[self.current_index])
        self.update_navigation_buttons()

    def update_navigation_buttons(self):
        self.backButton.setEnabled(self.current_index > 0)
        self.forwardButton.setEnabled(self.current_index < len(self.processed_images) - 1)


# Main application code
if __name__ == '__main__':
    app = QApplication([])
    window = ImageProcessor()
    window.show()
    app.exec_()
