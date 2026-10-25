import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple
from src.logger import logger

class ImageDamageAssessor:
    """
    Automated computer vision system to assess structural damage by comparing
    pre-disaster and post-disaster imagery.
    """
    
    @staticmethod
    def preprocess_image(image: Image.Image) -> np.ndarray:
        """Converts PIL Image to a normalized grayscale OpenCV matrix."""
        cv_img = np.array(image.convert("RGB"))
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred

    @classmethod
    def analyze_damage(cls, pre_image: Image.Image, post_image: Image.Image) -> Tuple[float, Image.Image, Dict[str, Any]]:
        """
        Compares pre and post imagery using structural change analysis, edge-density changes,
        and threshold difference masks.
        
        Returns:
            Tuple[float, Image.Image, Dict[str, Any]]:
                - damage_score (float): Calculated damage level between 0.0 (None) and 1.0 (Total destruction).
                - annotated_image (Image.Image): Post-disaster image with detected damage bounding boxes.
                - metrics (Dict): Breakdown of pixels changed, structural variance, and class.
        """
        logger.info("Initializing CV-based image damage analysis.")
        
        # Resize post-image to match pre-image size if they differ
        if pre_image.size != post_image.size:
            logger.info(f"Resizing post-image from {post_image.size} to match pre-image {pre_image.size}")
            post_image = post_image.resize(pre_image.size, Image.Resampling.LANCZOS)

        # Convert to CV format
        pre_cv = cls.preprocess_image(pre_image)
        post_cv = cls.preprocess_image(post_image)
        
        # Convert original post image to color for annotation
        annotated_cv = cv2.cvtColor(np.array(post_image.convert("RGB")), cv2.COLOR_RGB2BGR)

        # Compute Absolute Difference
        diff = cv2.absdiff(pre_cv, post_cv)
        
        # Threshold the difference image
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours of changes
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_pixels = pre_cv.shape[0] * pre_cv.shape[1]
        changed_pixels = 0
        damage_bounding_boxes = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter out tiny noise fluctuations
            if area > 150:
                changed_pixels += area
                damage_bounding_boxes += 1
                # Draw red bounding boxes around detected damage
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(annotated_cv, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
        # Calculate percentage change
        change_ratio = min(changed_pixels / total_pixels, 1.0)
        
        # Map change ratio to damage score using structural contrast calculation
        mean_diff = np.mean(diff) / 255.0
        damage_score = min(0.7 * change_ratio + 0.3 * mean_diff, 1.0)

        # Classification based on score limits
        if damage_score < 0.05:
            classification = "No Damage"
        elif damage_score < 0.25:
            classification = "Minor Damage"
        elif damage_score < 0.60:
            classification = "Major Damage"
        else:
            classification = "Destroyed"

        metrics = {
            "changed_area_ratio": float(change_ratio),
            "mean_intensity_diff": float(mean_diff),
            "detected_damage_sites": damage_bounding_boxes,
            "classification": classification
        }

        # Convert CV image back to PIL
        annotated_rgb = cv2.cvtColor(annotated_cv, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_rgb)
        
        logger.info(f"CV analysis completed: Damage Score = {damage_score:.4f}, Class = {classification}")
        return float(damage_score), annotated_pil, metrics


class GeospatialHazardDamageAssessor:
    """
    Predictive spatial modeling using physical vulnerability curves for
    hazards like floods or high wind speeds (hurricanes).
    """

    @staticmethod
    def calculate_wind_damage(wind_speed_mph: float, building_type: str = "residential") -> float:
        """
        Applies a HAZUS-like vulnerability curve to estimate structural damage.
        Vulnerability equation: damage_ratio = 1 / (1 + exp(-k * (speed - threshold)))
        """
        # Parameters representing susceptibility of building classes
        params = {
            "residential": {"k": 0.08, "v0": 74.0},  # hurricane category 1 threshold
            "commercial": {"k": 0.07, "v0": 85.0},
            "critical_infrastructure": {"k": 0.05, "v0": 100.0}
        }
        
        p = params.get(building_type.lower(), params["residential"])
        
        if wind_speed_mph < p["v0"]:
            return 0.0
            
        damage_ratio = 1.0 / (1.0 + np.exp(-p["k"] * (wind_speed_mph - p["v0"])))
        return float(np.clip(damage_ratio, 0.0, 1.0))

    @staticmethod
    def calculate_flood_damage(flood_depth_feet: float, building_type: str = "residential") -> float:
        """
        Applies standard FEMA depth-damage relationship parameters to estimate structural losses.
        """
        # Approximate FEMA depth-damage ratios for single-story structures
        # depths from -2 to 10 feet
        depth_points = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        
        damage_ratios = {
            "residential": [0.0, 0.0, 0.08, 0.20, 0.32, 0.44, 0.52, 0.60, 0.68, 0.73, 0.77, 0.80, 0.85],
            "commercial": [0.0, 0.0, 0.05, 0.12, 0.22, 0.35, 0.45, 0.53, 0.60, 0.65, 0.70, 0.75, 0.80],
            "critical_infrastructure": [0.0, 0.0, 0.02, 0.08, 0.15, 0.25, 0.35, 0.44, 0.52, 0.58, 0.64, 0.70, 0.75]
        }
        
        ratios = damage_ratios.get(building_type.lower(), damage_ratios["residential"])
        
        if flood_depth_feet <= depth_points[0]:
            return 0.0
        if flood_depth_feet >= depth_points[-1]:
            return ratios[-1]
            
        # Perform linear interpolation
        return float(np.interp(flood_depth_feet, depth_points, ratios))
