import cv2
import os
import time
from datetime import datetime
import threading
import numpy as np
import pyautogui

class TableScreenCapture:
    """Manages screen capture for the snooker table detection system"""
    
    def __init__(self, output_dir="captured_images", region=None):
        """
        Initialize the screen capture manager
        
        Args:
            output_dir (str): Directory to save captured images
            region (tuple): (x, y, width, height) region to capture, None for full screen
        """
        self.output_dir = output_dir
        self.region = region
        self.is_running = False
        self.preview_frame = None
        self.lock = threading.Lock()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def start(self):
        """Start the screen capture process"""
        if self.is_running:
            return
            
        # Start capture thread
        self.is_running = True
        self.thread = threading.Thread(target=self._update_frame)
        self.thread.daemon = True
        self.thread.start()
        
        print("Screen capture started successfully")
    
    def _update_frame(self):
        """Background thread to continuously update frames"""
        while self.is_running:
            try:
                # Capture screen
                screenshot = pyautogui.screenshot(region=self.region)
                # Convert to numpy array
                frame = np.array(screenshot)
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                with self.lock:
                    self.preview_frame = frame
            except Exception as e:
                print(f"Error capturing screen: {e}")
            
            time.sleep(0.1)  # ~10 fps for preview to reduce CPU usage
    
    def get_preview_frame(self):
        """Get the current preview frame for display"""
        with self.lock:
            if self.preview_frame is not None:
                return self.preview_frame.copy()
            return None
    
    def capture_image(self):
        """
        Capture the current screen and save it to disk
        
        Returns:
            str: Path to the saved image file
        """
        if not self.is_running:
            self.start()
        
        # Take a high-quality screenshot for the actual capture
        screenshot = pyautogui.screenshot(region=self.region)
        frame = np.array(screenshot)
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Generate a unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"table_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        # Save the image
        cv2.imwrite(filepath, frame)
        
        print(f"Screen captured and saved to {filepath}")
        return filepath
    
    def stop(self):
        """Stop the screen capture and release resources"""
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        print("Screen capture stopped")

# For backwards compatibility and simpler use
def capture_table_image(output_dir="captured_images", region=None):
    """
    Capture a screenshot of the desktop (or specified region)
    
    Args:
        output_dir (str): Directory to save captured images
        region (tuple): (x, y, width, height) region to capture, None for full screen
    
    Returns:
        str: Path to the saved image file
    """
    capture = TableScreenCapture(output_dir, region)
    try:
        return capture.capture_image()
    finally:
        capture.stop()
