import sys

import numpy as np
from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtCore import QTimer
import torch
import torch.nn as nn
import cv2
import torch.amp
from torchvision import transforms
from PIL import Image
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 🔥 CRITICAL
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sigma_range=(0,
             30)
sigma_mean = (sigma_range[1] - sigma_range[0]) / 2
sigma_std = (sigma_range[1] - sigma_range[0]) / 2
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


class WebcamViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Webcam Feed with Sharpness Score")

        # UI elements
        self.image_label = QLabel()
        self.sharpness_label = QLabel("Sharpness: ")
        self.sharpness_label.setFont(QFont("Arial", 14))

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.sharpness_label)
        self.setLayout(layout)

        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.model = BlurRegressionCNN().to(device)
        self.model.load_state_dict(torch.load("best_blur_model.pth", map_location=device))
        self.model.eval()

        # Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def preprocess_frame(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize to expected input size (e.g., 240x320)
        resized = cv2.resize(gray, (320, 240))
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        # Add batch and channel dimensions: (1, 1, H, W)
        tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(device)
        return tensor

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert BGR (OpenCV) to RGB (Qt)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qt_image))

            # Convert to PIL, grayscale
            img_pil = Image.fromarray(frame).convert("L")

            # Apply transform
            input_tensor = transform(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                sharpness_score = self.model(input_tensor).item()
                denormalized = sharpness_score * sigma_std + sigma_mean

            # Update label
            self.sharpness_label.setText(f"Sharpness: {denormalized:.2f}")

    def closeEvent(self, event):
        self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = WebcamViewer()
    viewer.show()
    sys.exit(app.exec())
