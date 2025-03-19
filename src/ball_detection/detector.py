import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter

class Ball:
    """Represents a snooker ball with its properties"""
    
    def __init__(self, color: str, position: Tuple[int, int], radius: float, confidence: float = 1.0):
        """
        Initialize a ball object
        
        Args:
            color: Ball color name (e.g. 'red', 'white', 'black')
            position: (x, y) coordinates of the ball center
            radius: Radius of the ball in pixels
            confidence: Detection confidence score (0-1)
        """
        self.color = color
        self.position = position
        self.radius = radius
        self.confidence = confidence
        
    def __repr__(self):
        return f"Ball(color='{self.color}', position={self.position}, radius={self.radius:.1f}, confidence={self.confidence:.2f})"


class BallDetector:
    """Detects snooker balls in an image using adaptive color analysis"""
    
    # Ball types in snooker
    BALL_TYPES = ['red', 'yellow', 'green', 'brown', 'blue', 'pink', 'black', 'white']
    
    # Maximum number of each ball type in a standard snooker game
    BALL_COUNTS = {
        'red': 15,
        'yellow': 1,
        'green': 1,
        'brown': 1,
        'blue': 1,
        'pink': 1,
        'black': 1,
        'white': 1
    }
    
    # Approximate BGR values for reference (not used directly for detection)
    REFERENCE_BGR = {
        'red': [40, 40, 200],     # Red ball
        'yellow': [30, 220, 220],  # Yellow ball
        'green': [30, 180, 30],    # Green ball
        'brown': [40, 80, 140],    # Brown ball
        'blue': [200, 60, 40],     # Blue ball  
        'pink': [180, 120, 220],   # Pink ball
        'black': [20, 20, 20],     # Black ball
        'white': [230, 230, 230]   # White ball
    }
    
    def __init__(self, 
                 min_ball_radius_ratio: float = 0.01,
                 max_ball_radius_ratio: float = 0.03,
                 circularity_threshold: float = 0.6,  # Reduced from 0.7 to allow less circular shapes
                 min_ball_area: int = 100,
                 debug: bool = False):
        """
        Initialize the ball detector
        
        Args:
            min_ball_radius_ratio: Minimum ball radius relative to table width
            max_ball_radius_ratio: Maximum ball radius relative to table width
            circularity_threshold: Minimum circularity for valid balls (0-1)
            min_ball_area: Minimum area for a valid ball in pixels
            debug: Whether to show debug images
        """
        self.min_ball_radius_ratio = min_ball_radius_ratio
        self.max_ball_radius_ratio = max_ball_radius_ratio
        self.circularity_threshold = circularity_threshold
        self.min_ball_area = min_ball_area
        self.debug = debug
        
    def detect_balls(self, image: np.ndarray, 
                    table_corners: List[Tuple[int, int]] = None) -> List[Ball]:
        """
        Detect all snooker balls in the image using multi-approach detection
        
        Args:
            image: BGR image containing snooker table and balls
            table_corners: Optional corner coordinates of the table for proportional sizing
            
        Returns:
            List of detected Ball objects
        """
        # Make a copy of the input image
        img = image.copy()
        
        # Calculate table dimensions and create mask
        table_width, table_mask, img_masked, sorted_corners = self._prepare_image(img, table_corners)
            
        # Calculate ball radius range based on table width
        min_radius = max(1, int(table_width * self.min_ball_radius_ratio))
        max_radius = max(min_radius + 5, int(table_width * self.max_ball_radius_ratio))
        
        # Use multiple approaches to find ball candidates
        candidates = []
        
        # Step 1: Use blob detection (more robust to reflections and soft edges)
        blob_candidates = self._detect_balls_via_blob(img_masked, min_radius, max_radius)
        candidates.extend(blob_candidates)
        
        # Step 2: Use color-based segmentation (better for balls with distinct colors)
        color_candidates = self._detect_balls_via_color_segmentation(img_masked, min_radius, max_radius, table_mask)
        
        # Merge candidates from different approaches
        merged_candidates = self._merge_candidates(candidates, color_candidates, min_radius)
        
        if not merged_candidates:
            return []
            
        if self.debug:
            self._debug_show_candidates(img, merged_candidates, "Ball Candidates")
        
        # Extract color features for each ball candidate
        ball_features = []
        for candidate in merged_candidates:
            features = self._extract_color_features(img, candidate)
            ball_features.append((candidate, features))
        
        # Perform color-based clustering and classification
        detected_balls = self._classify_balls_by_color(img, ball_features)
        
        # Filter balls (remove overlaps, apply count constraints)
        filtered_balls = self._filter_balls(detected_balls)
        
        # Visualization
        if self.debug:
            self._debug_show_balls(img, filtered_balls, sorted_corners, "Detected Balls")
        
        return filtered_balls
        
    def _prepare_image(self, img: np.ndarray, table_corners: List[Tuple[int, int]]) -> Tuple[float, np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """Prepare the image for processing and calculate table dimensions"""
        # Calculate table width if corners are provided
        table_width = None
        sorted_corners = None
        
        if table_corners and len(table_corners) >= 4:
            sorted_corners = self._sort_corners(table_corners)
            # Calculate width as distance between top-left and top-right corners
            top_left, top_right = sorted_corners[0], sorted_corners[1]
            table_width = np.sqrt((top_right[0] - top_left[0])**2 + 
                                  (top_right[1] - top_left[1])**2)
        
        # If table width not available, use image width
        if table_width is None:
            table_width = img.shape[1]
            
        # Create a mask for the table area if corners are provided
        table_mask = None
        if table_corners and len(table_corners) >= 4:
            table_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            cv2.fillPoly(table_mask, [np.array(sorted_corners, dtype=np.int32)], 255)
            img_masked = cv2.bitwise_and(img, img, mask=table_mask)
        else:
            table_mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
            img_masked = img.copy()
        
        return table_width, table_mask, img_masked, sorted_corners
    
    def _detect_balls_via_blob(self, img: np.ndarray, min_radius: int, max_radius: int) -> List[Dict]:
        """
        Detect potential ball candidates using blob detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing to improve ball visibility
        # - Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        if self.debug:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1), plt.imshow(gray, cmap='gray')
            plt.title('Grayscale'), plt.axis('off')
            plt.subplot(1, 3, 2), plt.imshow(blurred, cmap='gray')
            plt.title('Blurred'), plt.axis('off')
            plt.subplot(1, 3, 3), plt.imshow(enhanced, cmap='gray')
            plt.title('Enhanced (CLAHE)'), plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        # Set up the blob detector parameters
        params = cv2.SimpleBlobDetector_Params()
        
        # Change thresholds - lower min threshold to detect darker balls
        params.minThreshold = 10
        params.maxThreshold = 255
        
        # Filter by area
        params.filterByArea = True
        params.minArea = np.pi * (min_radius ** 2)
        params.maxArea = np.pi * (max_radius ** 2) * 1.5  # Bit larger to account for reflections
        
        # Filter by circularity
        params.filterByCircularity = True
        params.minCircularity = self.circularity_threshold - 0.1  # Lower threshold for blob detection
        
        # Filter by convexity
        params.filterByConvexity = True
        params.minConvexity = 0.8
        
        # Filter by inertia ratio (how elongated a shape is - 1.0 is a perfect circle)
        params.filterByInertia = True
        params.minInertiaRatio = 0.7
        
        # Create blob detector
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(enhanced)
        
        # Create candidate objects from keypoints
        candidates = []
        for kp in keypoints:
            center = (int(kp.pt[0]), int(kp.pt[1]))
            radius = int(kp.size / 2)  # size is diameter
            
            # Create a mask for this ball
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            
            candidates.append({
                'center': center,
                'radius': radius,
                'mask': mask,
                'circularity': 0.9,  # SimpleBlobDetector already filters by circularity
                'detection_type': 'blob'
            })
        
        if self.debug:
            # Draw detected blobs
            blob_img = cv2.drawKeypoints(img, keypoints, np.array([]), 
                                        (0, 255, 0), 
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            plt.figure(figsize=(10, 7))
            plt.imshow(cv2.cvtColor(blob_img, cv2.COLOR_BGR2RGB))
            plt.title(f'Blob Detection: {len(candidates)} candidates')
            plt.axis('off')
            plt.show()
            
        return candidates
    
    def _detect_balls_via_color_segmentation(self, img: np.ndarray, min_radius: int, max_radius: int, table_mask: np.ndarray) -> List[Dict]:
        """
        Detect potential ball candidates using color segmentation
        """
        candidates = []
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # We'll try to detect balls with distinctive colors (white, black, colored balls)
        # using color thresholds specific to each type of ball
        
        # List of color segmentation configurations (name, lower_hsv, upper_hsv)
        color_configs = [
            # White ball (high value, low saturation)
            ('white', np.array([0, 0, 180]), np.array([180, 50, 255])),
            
            # Black ball (low value)
            ('black', np.array([0, 0, 0]), np.array([180, 255, 50])),
            
            # Yellow ball
            ('yellow', np.array([20, 100, 100]), np.array([35, 255, 255])),
            
            # Blue ball
            ('blue', np.array([100, 70, 70]), np.array([130, 255, 255])),
            
            # Green ball
            ('green', np.array([40, 40, 40]), np.array([80, 255, 255])),
            
            # Pink ball
            ('pink', np.array([145, 30, 150]), np.array([165, 255, 255])),
            
            # Red balls (two ranges to handle the hue wrap-around)
            ('red1', np.array([0, 120, 70]), np.array([10, 255, 255])),
            ('red2', np.array([170, 120, 70]), np.array([180, 255, 255]))
        ]
        
        # Process each color range
        for color_name, lower, upper in color_configs:
            # Create mask for this color range
            color_mask = cv2.inRange(hsv, lower, upper)
            
            # Combine red masks if processing the second red range
            if color_name == 'red2':
                red1_mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
                color_mask = cv2.bitwise_or(color_mask, red1_mask)
                color_name = 'red'  # Rename for consistency
            
            # Apply table mask to restrict detection to table area
            if table_mask is not None:
                color_mask = cv2.bitwise_and(color_mask, table_mask)
            
            # Clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
            color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            if self.debug and np.sum(color_mask) > 0:
                plt.figure(figsize=(8, 6))
                plt.imshow(color_mask, cmap='gray')
                plt.title(f'{color_name} Color Mask')
                plt.axis('off')
                plt.show()
            
            # Find contours
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Skip small contours
                if area < self.min_ball_area:
                    continue
                
                # Check if area is within range for a ball
                min_area = np.pi * (min_radius ** 2)
                max_area = np.pi * (max_radius ** 2) * 1.5  # Allow a bit larger
                
                if area < min_area or area > max_area:
                    continue
                
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Skip non-circular objects
                if circularity < self.circularity_threshold:
                    continue
                
                # Find the minimum enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # Check if radius is within expected range
                if radius < min_radius or radius > max_radius:
                    continue
                
                # Create a mask for this ball
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                
                # Add to candidates
                candidates.append({
                    'center': (int(x), int(y)),
                    'radius': int(radius),
                    'mask': mask,
                    'circularity': circularity,
                    'detection_type': 'color_' + color_name,
                    'color_hint': color_name.replace('1', '').replace('2', '')  # Remove number suffixes
                })
        
        if self.debug:
            # Draw detected candidates
            color_img = img.copy()
            for candidate in candidates:
                cv2.circle(color_img, candidate['center'], candidate['radius'], (0, 255, 0), 2)
                
            plt.figure(figsize=(10, 7))
            plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
            plt.title(f'Color Segmentation: {len(candidates)} candidates')
            plt.axis('off')
            plt.show()
        
        return candidates
    
    def _merge_candidates(self, candidates1: List[Dict], candidates2: List[Dict], min_radius: int) -> List[Dict]:
        """
        Merge candidates from different detection approaches, removing duplicates
        """
        merged = candidates1.copy()
        
        # Check each candidate from the second list
        for c2 in candidates2:
            is_duplicate = False
            for c1 in candidates1:
                # Calculate distance between centers
                dist = np.sqrt((c1['center'][0] - c2['center'][0])**2 + 
                             (c1['center'][1] - c2['center'][1])**2)
                
                # If centers are close, they're the same ball
                if dist < min_radius * 1.5:
                    is_duplicate = True
                    break
            
            # Add if it's not a duplicate
            if not is_duplicate:
                merged.append(c2)
                
        return merged
    
    def _extract_color_features(self, img: np.ndarray, candidate: Dict) -> Dict:
        """Extract color features for a ball candidate, handling reflections"""
        # Get a slightly smaller region within the ball to avoid edge effects
        center = candidate['center']
        radius = int(candidate['radius'] * 0.8)  # Use 80% of radius to avoid edges
        x, y = center
        
        # Create a circular mask for the ball
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        
        # Apply mask to get just the ball pixels
        ball_region = cv2.bitwise_and(img, img, mask=mask)
        
        # Convert to different color spaces
        ball_hsv = cv2.cvtColor(ball_region, cv2.COLOR_BGR2HSV)
        ball_lab = cv2.cvtColor(ball_region, cv2.COLOR_BGR2LAB)
        
        # Create a mask to exclude reflections (very bright pixels)
        value_channel = ball_hsv[:,:,2]
        brightness_threshold = 220  # High value indicates reflection
        non_reflection_mask = np.logical_and(
            mask > 0,  # Ball area
            value_channel < brightness_threshold  # Not too bright (not reflection)
        ).astype(np.uint8) * 255
        
        # If too much of the ball is identified as reflection, adjust the threshold
        reflection_ratio = 1.0 - (np.sum(non_reflection_mask > 0) / np.sum(mask > 0))
        
        # If more than 70% of the ball is reflection, gradually decrease the threshold
        if reflection_ratio > 0.7:
            brightness_threshold = 240  # Allow brighter pixels
            non_reflection_mask = np.logical_and(
                mask > 0,
                value_channel < brightness_threshold
            ).astype(np.uint8) * 255
        
        # Get only the non-reflection ball pixels for color analysis
        bgr_pixels = img[non_reflection_mask > 0].reshape(-1, 3)
        hsv_pixels = ball_hsv[non_reflection_mask > 0].reshape(-1, 3)
        lab_pixels = ball_lab[non_reflection_mask > 0].reshape(-1, 3)
        
        # Handle case where most/all pixels are reflections
        if len(bgr_pixels) < 20:  # Too few non-reflection pixels
            # Fall back to using all pixels
            bgr_pixels = img[mask > 0].reshape(-1, 3)
            hsv_pixels = ball_hsv[mask > 0].reshape(-1, 3)
            lab_pixels = ball_lab[mask > 0].reshape(-1, 3)
        
        if len(bgr_pixels) == 0:  # No pixels found (shouldn't happen)
            return {
                'mean_bgr': np.array([0, 0, 0]),
                'mean_hsv': np.array([0, 0, 0]),
                'mean_lab': np.array([0, 0, 0]),
                'std_bgr': np.array([0, 0, 0]),
                'dominant_colors': np.array([[0, 0, 0]]),
                'dominant_colors_percent': np.array([1.0]),
                'color_consistency': 0.0
            }
        
        # Calculate mean color values, excluding outliers
        # Use quantiles to avoid extreme values that might be reflections
        mean_bgr = np.percentile(bgr_pixels, 50, axis=0)  # Median color
        std_bgr = np.std(bgr_pixels, axis=0)
        
        # We'll use HSV for better color analysis - take the most saturated regions
        # which typically represent the true ball color away from reflections
        saturation = hsv_pixels[:, 1]
        
        # Get the top 30% most saturated pixels (these are likely the clearest colors)
        if len(saturation) >= 10:  # Need enough pixels for percentile calculation
            sat_threshold = np.percentile(saturation, 70)  # Top 30%
            clear_color_mask = saturation >= sat_threshold
            
            if np.sum(clear_color_mask) >= 5:  # Need enough pixels for analysis
                hsv_clear = hsv_pixels[clear_color_mask]
                bgr_clear = bgr_pixels[clear_color_mask]
                
                # Use these clearer pixels for color analysis
                mean_hsv = np.mean(hsv_clear, axis=0)
                mean_bgr_clear = np.mean(bgr_clear, axis=0)
                
                # For white balls, saturation-based filtering might not work well
                # Check if the clear color is significantly different from the median
                color_diff = np.linalg.norm(mean_bgr_clear - mean_bgr)
                if color_diff > 50:  # Significant difference indicates we're filtering out too much
                    mean_bgr_clear = mean_bgr  # Fall back to median
            else:
                mean_hsv = np.mean(hsv_pixels, axis=0)
                mean_bgr_clear = mean_bgr
        else:
            mean_hsv = np.mean(hsv_pixels, axis=0)
            mean_bgr_clear = mean_bgr
        
        mean_lab = np.mean(lab_pixels, axis=0)
        
        # Find dominant colors using K-means clustering
        # Use fewer samples for efficiency
        num_samples = min(500, len(bgr_pixels))
        sample_indices = np.random.choice(len(bgr_pixels), num_samples, replace=False)
        bgr_samples = bgr_pixels[sample_indices]
        
        # Use at most 2 clusters to find dominant colors
        # (usually the main ball color and potentially the reflection)
        kmeans = KMeans(n_clusters=min(2, len(bgr_samples)), n_init=10)
        kmeans.fit(bgr_samples)
        
        # Get cluster centers (dominant colors) and their proportions
        dominant_colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        label_counts = Counter(labels)
        
        # Calculate percentage of each dominant color
        total_pixels = len(labels)
        dominant_colors_percent = [label_counts[i] / total_pixels for i in range(len(dominant_colors))]
        
        # Sort dominant colors by their percentage
        sorted_indices = np.argsort(dominant_colors_percent)[::-1]  # Descending order
        dominant_colors = dominant_colors[sorted_indices]
        dominant_colors_percent = np.array(dominant_colors_percent)[sorted_indices]
        
        # Calculate color consistency metric (higher is better)
        # Consistency is higher when dominant color represents more of the ball
        color_consistency = dominant_colors_percent[0] if len(dominant_colors_percent) > 0 else 0
        
        # If we found clearer colors, use those for the dominant color
        if 'mean_bgr_clear' in locals() and len(dominant_colors) > 0:
            # Replace the primary dominant color with the clear mean color
            dominant_colors[0] = mean_bgr_clear
        
        # Add color hint from detection if available
        color_hint = candidate.get('color_hint', None)
        
        return {
            'mean_bgr': mean_bgr,
            'mean_hsv': mean_hsv,
            'mean_lab': mean_lab, 
            'std_bgr': std_bgr,
            'dominant_colors': dominant_colors,
            'dominant_colors_percent': dominant_colors_percent,
            'color_consistency': color_consistency,
            'color_hint': color_hint
        }
    
    def _classify_balls_by_color(self, img: np.ndarray, ball_features: List[Tuple[Dict, Dict]]) -> List[Ball]:
        """Classify balls by color using clustering and adaptive color matching"""
        if not ball_features:
            return []
            
        # Extract the dominant color features for all balls
        dominant_colors = []
        color_hints = []
        
        for candidate, features in ball_features:
            if features['dominant_colors'].shape[0] > 0:
                # Use the most dominant color (already filtered for reflections)
                dominant_colors.append(features['dominant_colors'][0])
                color_hints.append(features.get('color_hint', None))
            else:
                # Fallback to mean color
                dominant_colors.append(features['mean_bgr'])
                color_hints.append(features.get('color_hint', None))
        
        dominant_colors = np.array(dominant_colors)
        
        # If we have too few balls, we can't reliably cluster
        if len(dominant_colors) < 3:
            # Use simple distance matching to reference colors
            return self._classify_by_color_distance(ball_features)
            
        # Determine the optimal number of clusters - should be at most the number of ball types
        # but also account for having multiple balls of the same color (e.g., red balls)
        num_clusters = min(len(self.BALL_TYPES), len(dominant_colors))
        
        # Cluster the colors using K-means
        kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        kmeans.fit(dominant_colors)
        
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # Match clusters to ball types using reference colors
        cluster_to_color = self._match_clusters_to_colors(cluster_centers, img, color_hints, labels)
        
        # Create ball objects
        balls = []
        for i, (candidate, features) in enumerate(ball_features):
            cluster_label = labels[i]
            # Use color hint if available and very confident, otherwise use cluster classification
            color_hint = features.get('color_hint', None)
            color_name = cluster_to_color[cluster_label]
            
            # Override with color hint if the hint is specific and we're very confident
            if color_hint in ['white', 'black', 'blue', 'pink', 'yellow']:  # These are distinctive
                # Color distance between dominant color and reference color for the hint
                ref_color = np.array(self.REFERENCE_BGR[color_hint])
                dominant_color = features['dominant_colors'][0] if features['dominant_colors'].shape[0] > 0 else features['mean_bgr']
                color_distance = np.linalg.norm(dominant_color - ref_color)
                
                # If the distance is reasonably small, use the hint
                if color_distance < 100:  # Threshold determined empirically
                    color_name = color_hint
            
            # Calculate confidence based on:
            # 1. Distance to cluster center (color match)
            # 2. Color consistency (how uniform the ball color is)
            # 3. Shape confidence (circularity)
            
            # Color distance confidence
            color_distance = np.linalg.norm(dominant_colors[i] - cluster_centers[cluster_label])
            max_distance = 255 * np.sqrt(3)  # Maximum possible distance in RGB space
            color_match_confidence = 1.0 - (color_distance / max_distance)
            
            # Color consistency confidence (from features)
            color_consistency = features.get('color_consistency', 0.5)  # Default to 0.5 if not calculated
            
            # Shape confidence
            shape_confidence = candidate.get('circularity', 0.8)
            
            # Combine all factors (weighted average)
            confidence = 0.4 * color_match_confidence + 0.3 * color_consistency + 0.3 * shape_confidence
            
            # Boost confidence if using color hint that matches the cluster assignment
            if color_hint is not None and color_hint == color_name:
                confidence = min(1.0, confidence + 0.1)
            
            balls.append(Ball(
                color=color_name,
                position=candidate['center'],
                radius=candidate['radius'],
                confidence=confidence
            ))
            
        return balls
    
    def _match_clusters_to_colors(self, cluster_centers: np.ndarray, img: np.ndarray = None, 
                                color_hints: List[str] = None, labels: np.ndarray = None) -> Dict[int, str]:
        """Match cluster centers to ball colors using reference colors and image-specific adjustments"""
        # For each cluster, find the closest reference color
        cluster_to_color = {}
        reference_colors = np.array([self.REFERENCE_BGR[color] for color in self.BALL_TYPES])
        
        # If we have the image, try to adjust reference colors based on lighting
        if img is not None:
            # Estimate global lighting by analyzing the table color (assume green)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv_img, (40, 40, 40), (80, 255, 255))  # Green table
            if np.sum(green_mask) > 0:
                green_bgr = cv2.mean(img, mask=green_mask)[:3]
                # If the table looks brighter/darker than expected, adjust reference colors
                brightness_factor = np.sum(green_bgr) / sum(self.REFERENCE_BGR['green'])
                if brightness_factor > 1.3 or brightness_factor < 0.7:  # Significant difference
                    # Adjust reference colors (except black and white which are already extremes)
                    adjusted_references = {}
                    for color in self.BALL_TYPES:
                        if color not in ['black', 'white']:
                            adjusted_references[color] = np.clip(
                                np.array(self.REFERENCE_BGR[color]) * brightness_factor,
                                0, 255
                            )
                        else:
                            adjusted_references[color] = self.REFERENCE_BGR[color]
                    
                    # Use adjusted references
                    reference_colors = np.array([adjusted_references[color] for color in self.BALL_TYPES])
        
        # Use color hints to improve cluster matching
        cluster_hint_counts = {}
        
        if color_hints and labels is not None:
            # Count how many of each hint are in each cluster
            for i, (hint, label) in enumerate(zip(color_hints, labels)):
                if hint is not None:
                    if label not in cluster_hint_counts:
                        cluster_hint_counts[label] = Counter()
                    cluster_hint_counts[label][hint] += 1
        
        # For each cluster, find the closest reference color
        for i, center in enumerate(cluster_centers):
            # Check if we have hints for this cluster
            if i in cluster_hint_counts:
                # Get the most common hint for this cluster
                most_common_hint = cluster_hint_counts[i].most_common(1)
                if most_common_hint:
                    hint, count = most_common_hint[0]
                    # If the hint appears enough times, use it directly
                    cluster_size = np.sum(labels == i)
                    if count / cluster_size > 0.5:  # More than half the cluster has the same hint
                        cluster_to_color[i] = hint
                        continue
            
            # Otherwise use distance-based matching
            distances = np.linalg.norm(reference_colors - center, axis=1)
            closest_color_idx = np.argmin(distances)
            closest_color = self.BALL_TYPES[closest_color_idx]
            cluster_to_color[i] = closest_color
        
        # Special handling for common detection issues:
        
        # 1. Ensure we have a white ball if we have anything detected
        # (white is sometimes hard to cluster correctly due to reflections)
        has_white = 'white' in cluster_to_color.values()
        
        if not has_white and len(cluster_centers) > 0:
            # Find the brightest cluster and mark it as white
            brightness = np.sum(cluster_centers, axis=1)
            brightest_cluster = np.argmax(brightness)
            
            # Only if it's bright enough (to avoid confusing black with white)
            if np.sum(cluster_centers[brightest_cluster]) > 450:  # Reasonably bright
                cluster_to_color[brightest_cluster] = 'white'
        
        # 2. Resolve conflicts when multiple clusters map to the same color
        # Particularly important for games with multiple red balls
        color_counts = Counter(cluster_to_color.values())
        
        # If a color appears more than it should (except red which can have many)
        for color, count in color_counts.items():
            if color != 'red' and count > self.BALL_COUNTS[color]:
                # Find all clusters assigned to this color
                problem_clusters = [c for c, col in cluster_to_color.items() if col == color]
                
                # Keep only the best matching one
                best_match = None
                min_distance = float('inf')
                
                for cluster in problem_clusters:
                    ref_color = np.array(self.REFERENCE_BGR[color])
                    distance = np.linalg.norm(cluster_centers[cluster] - ref_color)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_match = cluster
                
                # Reassign other clusters
                for cluster in problem_clusters:
                    if cluster != best_match:
                        # Find the next closest color for this cluster
                        center = cluster_centers[cluster]
                        distances = {}
                        
                        for ball_color in self.BALL_TYPES:
                            if ball_color != color:  # Skip the conflicting color
                                ref_color = np.array(self.REFERENCE_BGR[ball_color])
                                distances[ball_color] = np.linalg.norm(center - ref_color)
                        
                        # Select the next best match
                        next_best = min(distances.items(), key=lambda x: x[1])[0]
                        cluster_to_color[cluster] = next_best
        
        return cluster_to_color
    
    def _classify_by_color_distance(self, ball_features: List[Tuple[Dict, Dict]]) -> List[Ball]:
        """Classify balls by direct color distance when we have too few for clustering"""
        balls = []
        reference_colors = {color: np.array(bgr) for color, bgr in self.REFERENCE_BGR.items()}
        
        for candidate, features in ball_features:
            if features['dominant_colors'].shape[0] > 0:
                ball_color = features['dominant_colors'][0]
            else:
                ball_color = features['mean_bgr']
                
            # Find the closest reference color
            min_dist = float('inf')
            best_color = None
            
            for color_name, ref_color in reference_colors.items():
                dist = np.linalg.norm(ball_color - ref_color)
                if dist < min_dist:
                    min_dist = dist
                    best_color = color_name
            
            # Calculate confidence
            max_dist = 255 * np.sqrt(3)  # Maximum possible distance in RGB space
            color_confidence = 1.0 - (min_dist / max_dist)
            confidence = 0.7 * color_confidence + 0.3 * candidate['circularity']
            
            balls.append(Ball(
                color=best_color,
                position=candidate['center'],
                radius=candidate['radius'],
                confidence=confidence
            ))
            
        return balls
    
    def _filter_balls(self, balls: List[Ball]) -> List[Ball]:
        """
        Filter balls based on color counts and overlapping
        
        Args:
            balls: List of detected Ball objects
        
        Returns:
            Filtered list of Ball objects
        """
        # First, remove overlapping balls (prioritize by confidence)
        non_overlapping_balls = []
        sorted_balls = sorted(balls, key=lambda b: b.confidence, reverse=True)
        
        for ball in sorted_balls:
            # Check if this ball overlaps with any previously accepted ball
            overlapping = False
            for accepted_ball in non_overlapping_balls:
                # Calculate distance between centers
                dist = np.sqrt((ball.position[0] - accepted_ball.position[0])**2 + 
                               (ball.position[1] - accepted_ball.position[1])**2)
                
                # Balls overlap if distance is less than sum of radii
                if dist < (ball.radius + accepted_ball.radius) * 0.8:  # 0.8 factor allows slight overlap
                    overlapping = True
                    break
            
            if not overlapping:
                non_overlapping_balls.append(ball)
        
        # Group balls by color
        color_groups = {}
        for ball in non_overlapping_balls:
            if ball.color not in color_groups:
                color_groups[ball.color] = []
            color_groups[ball.color].append(ball)
        
        # Filter each color group
        filtered_balls = []
        for color, balls_group in color_groups.items():
            # Get the maximum allowed count for this color
            max_count = self.BALL_COUNTS.get(color, 0)
            
            if len(balls_group) <= max_count:
                # If we have fewer or equal balls than the maximum, keep them all
                filtered_balls.extend(balls_group)
            else:
                # If we have more balls than the maximum, keep the ones with highest confidence
                sorted_balls = sorted(balls_group, key=lambda b: b.confidence, reverse=True)
                filtered_balls.extend(sorted_balls[:max_count])
        
        return filtered_balls
    
    def _debug_show_candidates(self, img: np.ndarray, candidates: List[Dict], title: str):
        """Show debug visualization of ball candidates"""
        debug_img = img.copy()
        for candidate in candidates:
            cv2.circle(debug_img, candidate['center'], candidate['radius'], (0, 255, 0), 2)
        
        plt.figure(figsize=(10, 7))
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.title(f"{title}: {len(candidates)} candidates")
        plt.axis('off')
        plt.show()
    
    def _debug_show_balls(self, img: np.ndarray, balls: List[Ball], 
                          table_corners: List[Tuple[int, int]], title: str):
        """Show debug visualization of detected balls"""
        debug_img = img.copy()
        
        # Draw table outline if available
        if table_corners and len(table_corners) >= 4:
            pts = np.array(table_corners, dtype=np.int32)
            cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)
        
        # Draw each detected ball
        for ball in balls:
            x, y = ball.position
            r = int(ball.radius)
            
            # Set color based on ball type
            bgr_color = tuple(self.REFERENCE_BGR.get(ball.color, (128, 128, 128)))
            
            # Draw filled circle for the ball
            cv2.circle(debug_img, (x, y), r, bgr_color, -1)
            
            # Add outline
            outline_color = (0, 0, 0) if ball.color in ['white', 'yellow'] else (255, 255, 255)
            cv2.circle(debug_img, (x, y), r, outline_color, 2)
            
            # Add text with color name and confidence
            text = f"{ball.color} ({ball.confidence:.2f})"
            cv2.putText(debug_img, text, (x - r, y - r - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, outline_color, 2)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.title(f"{title}: {len(balls)} balls")
        plt.axis('off')
        plt.show()
    
    def _sort_corners(self, corners: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """
        Sort corners in clockwise order: top-left, top-right, bottom-right, bottom-left
        
        Args:
            corners: List of corner coordinates [(x1,y1), (x2,y2), ...]
            
        Returns:
            Sorted list of corners
        """
        if len(corners) < 4:
            return [(int(x), int(y)) for x, y in corners]
            
        # Calculate centroid
        cx = sum(x for x, y in corners) / len(corners)
        cy = sum(y for x, y in corners) / len(corners)
        
        # Convert to integers and sort by angle from centroid
        int_corners = [(int(x), int(y)) for x, y in corners]
        return sorted(int_corners, key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))


def detect_balls(image_path: str, table_corners: List[Tuple[int, int]] = None, 
                debug: bool = False):
    """
    Convenience function to detect balls in an image file
    
    Args:
        image_path: Path to the image file
        table_corners: Optional table corner coordinates
        debug: Whether to show debug output
        
    Returns:
        List of detected Ball objects
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Create detector and detect balls
    detector = BallDetector(debug=debug)
    balls = detector.detect_balls(img, table_corners)
    
    # Print detection results
    print(f"\nBall detection results for {image_path}:")
    print(f"Found {len(balls)} balls:")
    
    # Group balls by color for nicer output
    color_counts = {}
    for ball in balls:
        if ball.color not in color_counts:
            color_counts[ball.color] = 0
        color_counts[ball.color] += 1
    
    for color, count in color_counts.items():
        print(f"  {color}: {count}")
        
    # Print individual ball details
    for i, ball in enumerate(balls):
        print(f"  Ball {i+1}: {ball.color} at ({ball.position[0]}, {ball.position[1]}), radius={ball.radius:.1f}")
    
    return balls
