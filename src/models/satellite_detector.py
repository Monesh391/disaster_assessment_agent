import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from src.logger import setup_logger

logger = setup_logger("satellite_detector")

class SatelliteBuildingDetector:
    """
    Dedicated satellite building detector (YOLOv8-Small) trained on nadir overhead building footprints.
    Acts as the PRIMARY production building detector.
    Exposes identical interface to GroundingDINOModel for seamless, plug-and-play SAM2 compatibility.
    """
    def __init__(self, model_path="models/satellite_yolo_detector.pt", device="cpu"):
        self.device = device
        self.model_path = model_path
        
        # Absolute path resolution
        if not os.path.isabs(model_path):
            abs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), model_path)
            if os.path.exists(abs_path):
                self.model_path = abs_path
                
        logger.info(f"Initializing Primary Satellite Building Detector (YOLOv8-Small) from {self.model_path} on {device}...")
        
        if not os.path.exists(self.model_path):
            env_path = os.environ.get("SATELLITE_YOLO_PATH")
            if env_path and os.path.exists(env_path):
                self.model_path = env_path
            else:
                logger.warning(f"Trained weights not found at {self.model_path}, loading base yolov8s.pt")
                self.model_path = "yolov8s.pt"
                
        self.model = YOLO(self.model_path)
        logger.info(f"Primary Satellite Building Detector loaded successfully ({self.model_path}).")

    def detect_buildings(
        self,
        image: Image.Image,
        conf_threshold=0.30,
        iou_threshold=0.50,
        imgsz=1024
    ) -> torch.Tensor:
        """
        Executes building detection on the satellite image.
        Returns:
            bboxes: torch.Tensor of shape (N, 4) in normalized coordinates [cx, cy, w, h] in [0, 1]
        """
        logger.info(f"Executing Dedicated Satellite Detector inference (conf={conf_threshold}, iou={iou_threshold}, imgsz={imgsz})...")
        w_img, h_img = image.size
        
        # Run YOLO inference
        results = self.model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device=self.device,
            verbose=False
        )[0]
        
        norm_boxes = []
        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            for box in xyxy:
                x1, y1, x2, y2 = box
                x1 = max(0.0, min(float(w_img), x1))
                y1 = max(0.0, min(float(h_img), y1))
                x2 = max(0.0, min(float(w_img), x2))
                y2 = max(0.0, min(float(h_img), y2))
                
                w = (x2 - x1) / w_img
                h = (y2 - y1) / h_img
                cx = ((x1 + x2) / 2.0) / w_img
                cy = ((y1 + y2) / 2.0) / h_img
                
                if (x2 - x1) > 2 and (y2 - y1) > 2:
                    norm_boxes.append([cx, cy, w, h])
                    
        if not norm_boxes:
            return torch.zeros((0, 4), dtype=torch.float32)
        return torch.tensor(norm_boxes, dtype=torch.float32)

if __name__ == "__main__":
    detector = SatelliteBuildingDetector()
    test_img = Image.new("RGB", (1024, 1024), "gray")
    out = detector.detect_buildings(test_img)
    print(f"Output shape: {out.shape}")
