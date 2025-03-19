import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class TableDetector:
    """Detects the snooker table and its corner coordinates in an image"""
    
    def __init__(self, 
                 green_hue_range=(35, 100),  # Range of green hue values
                 green_saturation_range=(40, 255),  # Range of saturation for green
                 green_value_range=(40, 255),  # Range of value for green
                 edge_threshold1=100,  # Lower threshold for Canny edge detector
                 edge_threshold2=200,  # Upper threshold for Canny edge detector
                 hough_threshold=50,  # Threshold for Hough line detector
                 min_line_length=100,  # Minimum line length for Hough
                 max_line_gap=100,  # Maximum gap between line segments for Hough
                 debug=False):  # Whether to show debug images
        """
        Initialize the table detector with parameters for table detection
        
        Args:
            green_hue_range: Range of hue values for the green table (in HSV)
            green_saturation_range: Range of saturation values for the green table
            green_value_range: Range of brightness values for the green table
            edge_threshold1: Lower threshold for Canny edge detector
            edge_threshold2: Upper threshold for Canny edge detector
            hough_threshold: Threshold for Hough line detector
            min_line_length: Minimum line length for Hough line detector
            max_line_gap: Maximum gap between line segments for Hough
            debug: Whether to show debug images
        """
        self.green_hue_range = green_hue_range
        self.green_saturation_range = green_saturation_range
        self.green_value_range = green_value_range
        self.edge_threshold1 = edge_threshold1
        self.edge_threshold2 = edge_threshold2
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.debug = debug
    
    def detect_table(self, image: np.ndarray) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Detect the snooker table and its corner coordinates in the given image
        
        Args:
            image: OpenCV BGR image containing the snooker table
            
        Returns:
            corners: List of corner coordinates [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            mask: Binary mask of the detected table area
        """
        # Check if image is valid
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
        
        # Make a copy of the image
        img = image.copy()
        
        # Get image dimensions for relative positioning
        h, w = img.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create mask for green color
        lower_green = np.array([self.green_hue_range[0], 
                                self.green_saturation_range[0], 
                                self.green_value_range[0]])
        upper_green = np.array([self.green_hue_range[1], 
                                self.green_saturation_range[1], 
                                self.green_value_range[1]])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        # Find the largest contour (should be the table)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            if self.debug:
                print("No table contour found")
            return [], green_mask
        
        # Get the largest contour
        table_contour = max(contours, key=cv2.contourArea)
        
        # Create a mask with only the table contour
        table_mask = np.zeros_like(green_mask)
        cv2.drawContours(table_mask, [table_contour], 0, 255, -1)
        
        # Apply the mask to the original image
        masked_img = cv2.bitwise_and(img, img, mask=table_mask)
        
        # Convert to grayscale
        gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, self.edge_threshold1, self.edge_threshold2)
        
        # Show edges if debug is enabled
        if self.debug:
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 3, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Original Image'), plt.axis('off')
            plt.subplot(2, 3, 2), plt.imshow(green_mask, cmap='gray')
            plt.title('Green Mask'), plt.axis('off')
            plt.subplot(2, 3, 3), plt.imshow(table_mask, cmap='gray')
            plt.title('Table Mask'), plt.axis('off')
            plt.subplot(2, 3, 4), plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
            plt.title('Masked Image'), plt.axis('off')
            plt.subplot(2, 3, 5), plt.imshow(edges, cmap='gray')
            plt.title('Edges'), plt.axis('off')
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.hough_threshold, 
                                minLineLength=self.min_line_length, 
                                maxLineGap=self.max_line_gap)
        
        if lines is None or len(lines) == 0:
            if self.debug:
                print("No lines detected")
            return [], table_mask
        
        # Process lines - calculate length, angle, midpoint
        processed_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line length
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Calculate line angle (0-180 degrees)
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle > 180:
                angle = angle - 180
                
            # Calculate midpoint
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Line position relative to center
            position_x = "left" if mid_x < center_x else "right"
            position_y = "top" if mid_y < center_y else "bottom"
            
            # Line orientation
            orientation = "horizontal" if (angle < 45 or angle > 135) else "vertical"
            
            processed_lines.append({
                'coords': (x1, y1, x2, y2),
                'length': length,
                'angle': angle,
                'midpoint': (mid_x, mid_y),
                'position_x': position_x,
                'position_y': position_y,
                'orientation': orientation
            })
        
        # Group lines into horizontal and vertical for visualization
        horizontal_lines = [line['coords'] for line in processed_lines if line['orientation'] == 'horizontal']
        vertical_lines = [line['coords'] for line in processed_lines if line['orientation'] == 'vertical']
        
        # Draw lines if debug is enabled
        if self.debug:
            line_img = img.copy()
            for x1, y1, x2, y2 in horizontal_lines:
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for x1, y1, x2, y2 in vertical_lines:
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            plt.subplot(2, 3, 6), plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
            plt.title('Detected Lines'), plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        # Find the best lines for top, bottom, left, and right edges
        top_line = self._fit_line(processed_lines, 'top', center_x, center_y)
        bottom_line = self._fit_line(processed_lines, 'bottom', center_x, center_y)
        left_line = self._fit_line(processed_lines, 'left', center_x, center_y)
        right_line = self._fit_line(processed_lines, 'right', center_x, center_y)
        
        # Find intersections of the fitted lines to get the corners
        corners = []
        if top_line and left_line:
            corners.append(self._line_intersection(top_line, left_line))
        if top_line and right_line:
            corners.append(self._line_intersection(top_line, right_line))
        if bottom_line and left_line:
            corners.append(self._line_intersection(bottom_line, left_line))
        if bottom_line and right_line:
            corners.append(self._line_intersection(bottom_line, right_line))
        
        # Draw the fitted lines and corners if debug is enabled
        if self.debug:
            fitted_img = img.copy()
            
            # Draw fitted lines
            if top_line:
                x1, y1, x2, y2 = top_line
                cv2.line(fitted_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if bottom_line:
                x1, y1, x2, y2 = bottom_line
                cv2.line(fitted_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if left_line:
                x1, y1, x2, y2 = left_line
                cv2.line(fitted_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if right_line:
                x1, y1, x2, y2 = right_line
                cv2.line(fitted_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw corners
            for corner in corners:
                x, y = corner
                cv2.circle(fitted_img, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(fitted_img, cv2.COLOR_BGR2RGB))
            plt.title('Fitted Lines and Corners')
            plt.axis('off')
            plt.show()
        
        return corners, table_mask
    
    def _fit_line(self, lines: List[dict], mode: str, center_x: int, center_y: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Find the best line for a specific edge of the table
        
        Args:
            lines: List of processed line dictionaries
            mode: One of 'top', 'bottom', 'left', 'right'
            center_x: X-coordinate of the image center
            center_y: Y-coordinate of the image center
            
        Returns:
            Best line as (x1, y1, x2, y2) or None if no suitable line was found
        """
        if not lines:
            return None
        
        # Filter lines based on orientation and position
        filtered_lines = []
        
        if mode == 'top':
            # Looking for horizontal line in the top half
            filtered_lines = [
                line for line in lines 
                if line['orientation'] == 'horizontal' and line['position_y'] == 'top'
            ]
        elif mode == 'bottom':
            # Looking for horizontal line in the bottom half
            filtered_lines = [
                line for line in lines 
                if line['orientation'] == 'horizontal' and line['position_y'] == 'bottom'
            ]
        elif mode == 'left':
            # Looking for vertical line in the left half
            filtered_lines = [
                line for line in lines 
                if line['orientation'] == 'vertical' and line['position_x'] == 'left'
            ]
        elif mode == 'right':
            # Looking for vertical line in the right half
            filtered_lines = [
                line for line in lines 
                if line['orientation'] == 'vertical' and line['position_x'] == 'right'
            ]
        
        if not filtered_lines:
            # If no matching lines, try with just orientation
            if mode in ['top', 'bottom']:
                filtered_lines = [line for line in lines if line['orientation'] == 'horizontal']
            else:
                filtered_lines = [line for line in lines if line['orientation'] == 'vertical']
                
        if not filtered_lines:
            return None
            
        # Sort by length (longest first)
        filtered_lines.sort(key=lambda x: x['length'], reverse=True)
        
        # Get the longest line
        best_line = filtered_lines[0]['coords']
        
        # Extend the line to cover more of the table
        if mode in ['top', 'bottom']:
            x1, y1, x2, y2 = best_line
            # Make sure x1 is leftmost point
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
                
            # Calculate slope and intercept
            if x2 - x1 != 0:  # Avoid division by zero
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                
                # Extend to image edges
                new_x1 = max(0, center_x - 2000)  # Extend well beyond left edge
                new_x2 = min(center_x + 2000, center_x * 2)  # Extend well beyond right edge
                new_y1 = int(slope * new_x1 + intercept)
                new_y2 = int(slope * new_x2 + intercept)
                
                return (int(new_x1), int(new_y1), int(new_x2), int(new_y2))
            else:
                # Vertical line - just extend y coordinates
                return (x1, 0, x1, center_y * 2)
        else:  # left or right
            x1, y1, x2, y2 = best_line
            # Make sure y1 is topmost point
            if y1 > y2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
                
            # Calculate slope and intercept of the form x = my + b
            if y2 - y1 != 0:  # Avoid division by zero
                slope = (x2 - x1) / (y2 - y1)
                intercept = x1 - slope * y1
                
                # Extend to image edges
                new_y1 = max(0, center_y - 2000)  # Extend well beyond top edge
                new_y2 = min(center_y + 2000, center_y * 2)  # Extend well beyond bottom edge
                new_x1 = int(slope * new_y1 + intercept)
                new_x2 = int(slope * new_y2 + intercept)
                
                return (int(new_x1), int(new_y1), int(new_x2), int(new_y2))
            else:
                # Horizontal line - just extend x coordinates
                return (0, y1, center_x * 2, y1)
                
        return best_line
    
    def _line_intersection(self, line1: Tuple[int, int, int, int], 
                          line2: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Find the intersection point of two lines
        
        Args:
            line1: First line as (x1, y1, x2, y2)
            line2: Second line as (x1, y1, x2, y2)
            
        Returns:
            Intersection point as (x, y)
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        # Line 1 as ax + by + c = 0
        a1 = y2 - y1
        b1 = x1 - x2
        c1 = x2 * y1 - x1 * y2
        
        # Line 2 as ax + by + c = 0
        a2 = y4 - y3
        b2 = x3 - x4
        c2 = x4 * y3 - x3 * y4
        
        # Determinant
        det = a1 * b2 - a2 * b1
        
        if det == 0:
            # Lines are parallel, return midpoint
            return ((x1 + x2 + x3 + x4) / 4, (y1 + y2 + y3 + y4) / 4)
        
        # Calculate intersection
        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det
        
        return (x, y)

def detect_table_corners(image_path: str, debug: bool = False) -> List[Tuple[int, int]]:
    """
    Detect the snooker table corners in the given image
    
    Args:
        image_path: Path to the image file
        debug: Whether to show debug images
        
    Returns:
        List of corner coordinates [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Create a table detector and detect the table
    detector = TableDetector(debug=debug)
    corners, _ = detector.detect_table(img)
    
    # Print details about detection
    print(f"\nTable detection results for {image_path}:")
    print(f"Found {len(corners)} corner points")
    
    # Sort corners in clockwise order starting from top-left
    if len(corners) == 4:
        # Calculate centroid
        cx = sum(x for x, y in corners) / 4
        cy = sum(y for x, y in corners) / 4
        
        # Sort corners by angle from centroid
        corners = sorted(corners, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
        
        print("Corners (sorted clockwise from top-left):")
        for i, (x, y) in enumerate(corners):
            corner_type = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"][i]
            print(f"  {corner_type}: ({int(x)}, {int(y)})")
    else:
        print("Corners (unsorted):")
        for i, (x, y) in enumerate(corners):
            print(f"  Point {i+1}: ({int(x)}, {int(y)})")
    
    return corners
