import sys
import cv2
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
)
from PySide6.QtGui import QImage, QPixmap


class WebcamWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Webcam Feed")
        self.setGeometry(100, 100, 640, 480)

        # Create main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Create label to display video
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)

        # Create stop button
        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.close)
        self.layout.addWidget(self.stop_button)

        # Initialize webcam
        self.capture = cv2.VideoCapture(0)  # 0 is usually the default webcam
        if not self.capture.isOpened():
            print("Error: Could not open webcam")
            sys.exit()

        # Set up timer for updating video feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every ~30ms (~33fps)

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            # Convert OpenCV BGR image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to QImage
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # Scale image to fit label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(image)
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

            # Display image
            self.video_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        # Release webcam when closing
        self.capture.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = WebcamWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()