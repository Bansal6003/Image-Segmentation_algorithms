import cv2
import numpy as np
import os
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget, QMessageBox, QHBoxLayout, QInputDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO
from scipy import stats

class ImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Size Prediction')
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

        self.buttonsLayout = QHBoxLayout()
        layout.addLayout(self.buttonsLayout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.images = []
        self.image_paths = []
        self.processed_images = []
        self.contours_data = []
        self.class_buttons = []
        self.models = {}
        self.current_index = -1
        self.current_model = None
        self.thresholds = {}  # Dictionary to hold thresholds for each class

    def load_models(self):
        # Load all models
        model_paths = {
            'Eye': r'D:\Behavioral genetics_V1\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Collective_Trained_models_noBB_with_BB_ZB_images\train_eye_with_ZB_BB_images_noBB\weights\last.pt',
            # 'Head': r'C:\Users\Pushkar Bansal\Desktop\Behavioral genetics\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Trained_aug_models_noBB\train_aug_head_noBB\weights\last.pt',
            # 'Head-Yolk Extension': r'C:\Users\Pushkar Bansal\Desktop\Behavioral genetics\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Trained_aug_models_noBB\train_aug_head_yolkext_noBB\weights\last.pt',
            # 'Whole Larva': r'D:\Behavioral genetics_V1\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Collective_Trained_models_noBB_with_BB_ZB_images\train_whole_larva_with_ZB_BB_images_noBB\weights\best.pt',
            # 'Yolk-extension': r'C:\Users\Pushkar Bansal\Desktop\Behavioral genetics\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Trained_aug_models_noBB\train3_48_params_yolext_noBB\weights\last.pt',
            # 'Yolkext-tail': r'D:\Behavioral genetics_V1\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Collective_Trained_models_noBB_with_BB_ZB_images\train_yolkext_tail_with_ZB_BB_images_noBB\weights\last.pt',
            # 'Yolk-Sac': r'D:\Behavioral genetics_V1\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Yolov9_V2_WT_MIB_sampled\train_yolk_sac_WT_MIB_samples\weights\last.pt',
            # 'Pericardium': r'D:\Behavioral genetics_V1\AI_Project_Python_Templates\Annotation for yolo and coco\runs\segment\Collective_Trained_models_noBB_with_BB_ZB_images\train_pericardium_with_ZB_BB_images_noBB\weights\best.pt'
        }

        for class_name, model_path in model_paths.items():
            self.models[class_name] = YOLO(model_path)

    def upload_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder Containing Test Images')
        if not folder_path:
            return
    
        self.load_models()
    
        pixel_size = 0.008  # based on real-world dimensions
    
        self.images = []
        self.image_paths = []
        self.processed_images = []
        self.contours_data = []
        self.current_index = -1
    
        all_contours = {class_name: [] for class_name in self.models.keys()}
    
        # Create an output directory
        output_directory = os.path.join(folder_path, "processed_images")
        os.makedirs(output_directory, exist_ok=True)
    
        # Initialize DataFrame with size and anomaly columns for each class
        self.df = pd.DataFrame(columns=["Image"] + [f'{class_name}_Size' for class_name in self.models.keys()] + [f'{class_name}_Anomaly' for class_name in self.models.keys()])
    
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                image_path = os.path.join(folder_path, filename)
                original_image = cv2.imread(image_path)
                if original_image is not None:
                    self.images.append(original_image)
                    self.image_paths.append(image_path)
    
                    self.process_all_classes(image_path, pixel_size, all_contours, output_directory)
    
        # Compute thresholds based on IQR
        self.thresholds = self.compute_thresholds(all_contours)
    
        # Process again with thresholds and save results
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                image_path = os.path.join(folder_path, filename)
                contour_sizes = self.process_all_classes(image_path, pixel_size, all_contours, output_directory, True)
                self.df = pd.concat([self.df, contour_sizes])
    
        output_file = os.path.join(folder_path, "contour_sizes.xlsx")
        self.df.to_excel(output_file, index=False)
    
        if self.images:
            self.current_index = 0
            self.display_image(self.images[self.current_index])
            self.update_buttons()
            self.create_class_buttons()
    
            QMessageBox.information(self, 'Success', f'Contour sizes saved to {output_file}.')
        else:
            QMessageBox.warning(self, 'Warning', 'No images processed successfully.')


    def process_all_classes(self, image_path, pixel_size, all_contours, output_directory, include_anomaly=False):
        contour_sizes = {"Image": os.path.basename(image_path)}
    
        image_name = os.path.basename(image_path)
        image_without_extension = os.path.splitext(image_name)[0]
    
        has_anomaly = False
        anomaly_classes = []
    
        for class_name, model in self.models.items():
            contours_info = self.process_image(image_path, model, pixel_size)
            total_area = sum([info['area'] for info in contours_info])
            all_contours[class_name].extend([info['area'] for info in contours_info])
    
            # Separate size and anomaly status into different columns
            contour_sizes[class_name + '_Size'] = total_area
    
            if include_anomaly:
                lower_threshold, upper_threshold = self.thresholds.get(class_name, (float('-inf'), float('inf')))
                if total_area < lower_threshold or total_area > upper_threshold:
                    has_anomaly = True
                    anomaly_classes.append(class_name)
                    contour_sizes[class_name + '_Anomaly'] = 'Anomaly'
                else:
                    contour_sizes[class_name + '_Anomaly'] = 'No Anomaly'
            else:
                contour_sizes[class_name + '_Anomaly'] = 'No Anomaly'  # Default to 'No Anomaly'
    
        if include_anomaly:
            if has_anomaly:
                for class_name in anomaly_classes:
                    anomaly_directory = os.path.join(output_directory, class_name)
                    os.makedirs(anomaly_directory, exist_ok=True)
                    cv2.imwrite(os.path.join(anomaly_directory, image_name), cv2.imread(image_path))
            else:
                non_anomalous_directory = os.path.join(output_directory, "No_Anomalies")
                os.makedirs(non_anomalous_directory, exist_ok=True)
                cv2.imwrite(os.path.join(non_anomalous_directory, image_name), cv2.imread(image_path))
    
        return pd.DataFrame([contour_sizes])


    def process_image(self, image_path, model, pixel_size):
        image = cv2.imread(image_path)
    
        if image is None:
            print(f"Error loading image: {image_path}")
            return []
    
        results = model.predict(source=image_path, imgsz=640, conf=0.01, iou=0.2)
    
        contours_info = []
    
        for result in results:
            if result.masks is None:
                continue
    
            segments = result.masks.xy
            names = result.names
    
            for i, segment in enumerate(segments):
                class_id = int(result.boxes[i].cls.cpu().numpy()[0])
                class_name = names[class_id]
    
                contour = np.array(segment, dtype=np.int32).reshape((-1, 1, 2))
    
                if contour.size == 0:
                    print(f"Empty contour for class {class_name}")
                    continue
    
                # Adjust epsilon to control the degree of contour smoothing
                epsilon = 0.0035 * cv2.arcLength(contour, True)
                smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
    
                if smoothed_contour.shape[0] >= 3:
                    # Calculate the area of the smoothed contour
                    area = cv2.contourArea(smoothed_contour) * (pixel_size ** 2)
                    contours_info.append({
                        'class_name': class_name,
                        'contour': smoothed_contour,
                        'area': area
                    })
                else:
                    print(f"Invalid contour shape for class {class_name}")
    
        return contours_info


    def compute_thresholds(self, all_contours):
        thresholds = {}
        for class_name, areas in all_contours.items():
            if len(areas) > 0:
                q1, q3 = np.percentile(areas, [20, 60])
                iqr = q3 - q1
                lower_threshold = q1 - 1.5 * iqr
                upper_threshold = q3 + 1.5 * iqr
                thresholds[class_name] = (lower_threshold, upper_threshold)
        return thresholds
    

    def show_class_contours(self, class_name):
        if not self.images:
            return
    
        self.current_model = self.models[class_name]
    
        image_path = self.image_paths[self.current_index]
        contours_info = self.process_image(image_path, self.current_model, 0.008)
    
        original_image = self.images[self.current_index].copy()
    
        for contour_info in contours_info:
            contour = contour_info['contour']
            area = contour_info['area']
    
            if contour.shape[0] >= 3:
                cv2.drawContours(original_image, [contour], -1, (0, 255, 0), 2)
    
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
    
                    lower_threshold, upper_threshold = self.thresholds.get(class_name, (float('-inf'), float('inf')))
                    if area < lower_threshold or area > upper_threshold:
                        cv2.putText(original_image, f'{class_name} (anomaly)', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        cv2.putText(original_image, class_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
        self.display_image(original_image)

    
    def display_image(self, image):
        # Get the dimensions of the window
        window_width = self.label.width()
        window_height = self.label.height()
        
        # Resize the image to fit within the window dimensions, maintaining the aspect ratio
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        aspect_ratio = w / h
    
        if w > window_width or h > window_height:
            if aspect_ratio > 1:
                # Image is wider than tall, resize based on width
                new_width = window_width
                new_height = int(window_width / aspect_ratio)
            else:
                # Image is taller than wide, resize based on height
                new_height = window_height
                new_width = int(window_height * aspect_ratio)
        else:
            new_width, new_height = w, h
    
        resized_image = cv2.resize(image_rgb, (new_width, new_height))
    
        # Convert to QImage and display
        bytes_per_line = ch * new_width
        q_img = QImage(resized_image.data, new_width, new_height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.label.setPixmap(pixmap)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(True)

    def show_previous_image(self):
        if self.images and self.current_index > 0:
            self.current_index -= 1
            self.display_image(self.images[self.current_index])
            self.update_buttons()

    def show_next_image(self):
        if self.images and self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.display_image(self.images[self.current_index])
            self.update_buttons()

    def update_buttons(self):
        self.backButton.setEnabled(self.current_index > 0)
        self.forwardButton.setEnabled(self.current_index < len(self.images) - 1)

    def create_class_buttons(self):
        for class_name in self.models.keys():
            button = QPushButton(class_name, self)
            button.clicked.connect(lambda checked, name=class_name: self.show_class_contours(name))
            self.buttonsLayout.addWidget(button)
            self.class_buttons.append(button)

if __name__ == '__main__':
    app = QApplication([])
    window = ImageProcessor()
    window.show()
    app.exec_()
