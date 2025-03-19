import sys
import cv2
import numpy as np
import os
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QHBoxLayout, QWidget, QLabel, QComboBox, QCheckBox, 
                            QSpinBox, QGroupBox, QFormLayout, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QTimer
from image_processing.detector import TableScreenCapture
from table_detection.detector import TableDetector
from ball_detection.detector import BallDetector

class SnookerHawkEyeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snooker Hawk-Eye")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize screen capture
        self.capture_manager = TableScreenCapture()
        self.capture_manager.start()
        self.use_region = False
        self.region = (0, 0, 1920, 1080)  # Default region (x, y, width, height)
        
        # Initialize table detector and ball detector
        self.table_detector = TableDetector(debug=False)
        self.ball_detector = BallDetector(debug=True)
        self.table_corners = []
        self.detected_balls = []
        self.latest_captured_image = None
        
        # Create UI
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        self.central_widget.setLayout(main_layout)
        
        # Preview area
        self.preview_label = QLabel("Preview will appear here")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(800, 600)
        self.preview_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        main_layout.addWidget(self.preview_label)
        
        # Controls group
        controls_group = QGroupBox("Capture Controls")
        controls_layout = QVBoxLayout()
        controls_group.setLayout(controls_layout)
        main_layout.addWidget(controls_group)
        
        # Region selection
        region_layout = QHBoxLayout()
        controls_layout.addLayout(region_layout)
        
        self.region_checkbox = QCheckBox("Capture specific region")
        self.region_checkbox.stateChanged.connect(self.toggle_region_capture)
        region_layout.addWidget(self.region_checkbox)
        
        # Region controls
        region_form = QFormLayout()
        region_layout.addLayout(region_form)
        
        self.x_spin = QSpinBox()
        self.x_spin.setRange(0, 3000)
        self.x_spin.setValue(0)
        self.x_spin.setEnabled(False)
        region_form.addRow("X:", self.x_spin)
        
        self.y_spin = QSpinBox()
        self.y_spin.setRange(0, 3000)
        self.y_spin.setValue(0)
        self.y_spin.setEnabled(False)
        region_form.addRow("Y:", self.y_spin)
        
        self.width_spin = QSpinBox()
        self.width_spin.setRange(100, 3000)
        self.width_spin.setValue(1920)
        self.width_spin.setEnabled(False)
        region_form.addRow("Width:", self.width_spin)
        
        self.height_spin = QSpinBox()
        self.height_spin.setRange(100, 3000)
        self.height_spin.setValue(1080)
        self.height_spin.setEnabled(False)
        region_form.addRow("Height:", self.height_spin)
        
        # Button layout
        button_layout = QHBoxLayout()
        controls_layout.addLayout(button_layout)
        
        # Capture button
        self.capture_button = QPushButton("Capture Screenshot")
        self.capture_button.setMinimumHeight(50)
        self.capture_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.capture_button.clicked.connect(self.capture_image)
        button_layout.addWidget(self.capture_button)
        
        # Detect table button
        self.detect_button = QPushButton("Detect Table")
        self.detect_button.setMinimumHeight(50)
        self.detect_button.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.detect_button.clicked.connect(self.detect_table)
        self.detect_button.setEnabled(False)  # Disable until image is captured
        button_layout.addWidget(self.detect_button)
        
        # Detect balls button
        self.detect_balls_button = QPushButton("Detect Balls")
        self.detect_balls_button.setMinimumHeight(50)
        self.detect_balls_button.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold;")
        self.detect_balls_button.clicked.connect(self.detect_balls)
        self.detect_balls_button.setEnabled(False)  # Disable until table is detected
        button_layout.addWidget(self.detect_balls_button)
        
        # Status label
        self.status_label = QLabel("Ready. Press the button to capture the screen.")
        main_layout.addWidget(self.status_label)
        
        # Set up timer for updating preview
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_preview)
        self.timer.start(100)  # Update every 100ms (10 fps) to reduce CPU usage
    
    def toggle_region_capture(self, state):
        """Toggle capturing specific region"""
        self.use_region = (state == Qt.Checked)
        self.x_spin.setEnabled(self.use_region)
        self.y_spin.setEnabled(self.use_region)
        self.width_spin.setEnabled(self.use_region)
        self.height_spin.setEnabled(self.use_region)
        
        if self.use_region:
            self.update_region()
        else:
            self.capture_manager.region = None
    
    def update_region(self):
        """Update the region to capture"""
        self.region = (
            self.x_spin.value(),
            self.y_spin.value(),
            self.width_spin.value(),
            self.height_spin.value()
        )
        self.capture_manager.region = self.region
    
    def update_preview(self):
        """Update the screen preview"""
        frame = self.capture_manager.get_preview_frame()
        if frame is not None:
            # Convert to Qt format and display
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.preview_label.setPixmap(pixmap.scaled(
                self.preview_label.width(), 
                self.preview_label.height(),
                Qt.KeepAspectRatio
            ))
    
    def capture_image(self):
        """Capture a screenshot"""
        if self.use_region:
            self.update_region()
            
        try:
            filepath = self.capture_manager.capture_image()
            self.latest_captured_image = cv2.imread(filepath)
            self.status_label.setText(f"Screenshot captured: {filepath}")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.detect_button.setEnabled(True)
        except Exception as e:
            self.status_label.setText(f"Error capturing screenshot: {str(e)}")
            self.status_label.setStyleSheet("color: red;")
            self.detect_button.setEnabled(False)
    
    def detect_table(self):
        """Detect table corners in the captured image"""
        if self.latest_captured_image is None:
            QMessageBox.warning(self, "Warning", "No image captured yet. Please capture an image first.")
            return
            
        try:
            # Detect table corners
            corners, _ = self.table_detector.detect_table(self.latest_captured_image)
            self.table_corners = corners
            
            if corners:
                # Print corner coordinates to command line
                print("\n==== DETECTED TABLE CORNER COORDINATES ====")
                
                # Sort corners in clockwise order if we have 4 corners
                if len(corners) == 4:
                    cx = sum(x for x, y in corners) / 4
                    cy = sum(y for x, y in corners) / 4
                    sorted_corners = sorted(corners, 
                                           key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
                    
                    # Print sorted corners
                    for i, (x, y) in enumerate(sorted_corners):
                        print(f"Corner {i+1}: ({int(x)}, {int(y)})")
                        
                    print("\nCorner mapping:")
                    print("Corner 1: Top-left")
                    print("Corner 2: Top-right")
                    print("Corner 3: Bottom-right")
                    print("Corner 4: Bottom-left")
                else:
                    # Print unsorted corners
                    for i, (x, y) in enumerate(corners):
                        print(f"Corner {i+1}: ({int(x)}, {int(y)})")
                    
                    print(f"\nWarning: Found {len(corners)} corners instead of the expected 4.")
                    
                print("=========================================\n")
                
                # Display the corners on the image
                img_with_corners = self.latest_captured_image.copy()
                
                # Draw the corners
                for i, (x, y) in enumerate(corners):
                    cv2.circle(img_with_corners, (int(x), int(y)), 10, (0, 0, 255), -1)
                    cv2.putText(img_with_corners, f"{i+1}", (int(x)+15, int(y)+15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Draw lines between corners if we have 4 corners
                if len(corners) == 4:
                    # Sort corners in clockwise order
                    cx = sum(x for x, y in corners) / 4
                    cy = sum(y for x, y in corners) / 4
                    sorted_corners = sorted(corners, 
                                           key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
                    
                    # Draw lines
                    for i in range(4):
                        pt1 = (int(sorted_corners[i][0]), int(sorted_corners[i][1]))
                        pt2 = (int(sorted_corners[(i+1)%4][0]), int(sorted_corners[(i+1)%4][1]))
                        cv2.line(img_with_corners, pt1, pt2, (0, 255, 0), 2)
                
                # Display the result
                rgb_image = cv2.cvtColor(img_with_corners, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                self.preview_label.setPixmap(pixmap.scaled(
                    self.preview_label.width(), 
                    self.preview_label.height(),
                    Qt.KeepAspectRatio
                ))

                
                # Save the image with detected corners
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                detected_filename = f"table_detected_{timestamp}.jpg"
                detected_filepath = os.path.join("detected_images", detected_filename)
                
                # Create directory if it doesn't exist
                os.makedirs("detected_images", exist_ok=True)
                
                # Save the image
                cv2.imwrite(detected_filepath, img_with_corners)
                print(f"Image with detected corners saved to {detected_filepath}")
                
                # Update status
                self.status_label.setText(f"Table detected with {len(corners)} corners")
                self.status_label.setStyleSheet("color: blue; font-weight: bold;")
                
                # Enable ball detection if table was detected
                if corners:
                    self.detect_balls_button.setEnabled(True)
                else:
                    self.detect_balls_button.setEnabled(False)
            else:
                self.status_label.setText("No table corners detected. Try adjusting the capture region.")
                self.status_label.setStyleSheet("color: orange; font-weight: bold;")
                
        except Exception as e:
            self.status_label.setText(f"Error detecting table: {str(e)}")
            self.status_label.setStyleSheet("color: red;")
            self.detect_balls_button.setEnabled(False)
    
    def detect_balls(self):
        """Detect balls on the table"""
        if self.latest_captured_image is None:
            QMessageBox.warning(self, "Warning", "No image captured yet. Please capture an image first.")
            return
            
        if not self.table_corners:
            QMessageBox.warning(self, "Warning", "Table corners not detected. Please detect the table first.")
            return
            
        try:
            # Detect balls
            self.detected_balls = self.ball_detector.detect_balls(
                self.latest_captured_image, 
                self.table_corners
            )
            
            # Draw balls on the image
            img_with_balls = self.latest_captured_image.copy()
            
            # Draw table outline first
            if len(self.table_corners) >= 4:
                cx = sum(x for x, y in self.table_corners) / 4
                cy = sum(y for x, y in self.table_corners) / 4
                sorted_corners = sorted(self.table_corners, 
                                       key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
                
                # Draw table outline
                for i in range(4):
                    pt1 = (int(sorted_corners[i][0]), int(sorted_corners[i][1]))
                    pt2 = (int(sorted_corners[(i+1)%4][0]), int(sorted_corners[(i+1)%4][1]))
                    cv2.line(img_with_balls, pt1, pt2, (0, 255, 0), 2)
            
            # Draw the balls
            for ball in self.detected_balls:
                x, y = ball.position
                r = int(ball.radius)
                
                # Set color based on ball type
                if ball.color == 'red':
                    color = (0, 0, 255)  # BGR: Red
                elif ball.color == 'yellow':
                    color = (0, 255, 255)  # BGR: Yellow
                elif ball.color == 'green':
                    color = (0, 255, 0)  # BGR: Green
                elif ball.color == 'brown':
                    color = (42, 42, 165)  # BGR: Brown
                elif ball.color == 'blue':
                    color = (255, 0, 0)  # BGR: Blue
                elif ball.color == 'pink':
                    color = (147, 20, 255)  # BGR: Pink
                elif ball.color == 'black':
                    color = (0, 0, 0)  # BGR: Black
                elif ball.color == 'white':
                    color = (255, 255, 255)  # BGR: White
                else:
                    color = (128, 128, 128)  # BGR: Gray for unknown
                
                # Draw filled circle for the ball
                cv2.circle(img_with_balls, (x, y), r, color, -1)
                
                # Draw ball outline
                outline_color = (0, 0, 0) if ball.color == 'white' else (255, 255, 255)
                cv2.circle(img_with_balls, (x, y), r, outline_color, 2)
                
                # Add ball color text
                cv2.putText(img_with_balls, ball.color, (x - r, y - r - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, outline_color, 2)
            
            # Display the result
            rgb_image = cv2.cvtColor(img_with_balls, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.preview_label.setPixmap(pixmap.scaled(
                self.preview_label.width(), 
                self.preview_label.height(),
                Qt.KeepAspectRatio
            ))
            
            # Save the image with detected balls
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detected_filename = f"balls_detected_{timestamp}.jpg"
            detected_filepath = os.path.join("detected_images", detected_filename)
            
            # Create directory if it doesn't exist
            os.makedirs("detected_images", exist_ok=True)
            
            # Save the image
            cv2.imwrite(detected_filepath, img_with_balls)
            print(f"Image with detected balls saved to {detected_filepath}")
            
            # Print ball detection results
            print(f"Detected {len(self.detected_balls)} balls:")
            color_counts = {}
            for ball in self.detected_balls:
                if ball.color not in color_counts:
                    color_counts[ball.color] = 0
                color_counts[ball.color] += 1
                
            for color, count in color_counts.items():
                print(f"  {color}: {count}")
            
            # Update status
            self.status_label.setText(f"Detected {len(self.detected_balls)} balls ({', '.join(f'{c}:{n}' for c, n in color_counts.items())})")
            self.status_label.setStyleSheet("color: purple; font-weight: bold;")
            
        except Exception as e:
            self.status_label.setText(f"Error detecting balls: {str(e)}")
            self.status_label.setStyleSheet("color: red;")

    def closeEvent(self, event):
        """Clean up resources when closing the application"""
        self.timer.stop()
        self.capture_manager.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = SnookerHawkEyeApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
