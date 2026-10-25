import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from src.logger import logger
from transformers import Sam2Model, Sam2Processor

class SAM2Model:
    """
    Unified model wrapper for the official Segment Anything Model 2 (SAM2).
    Utilizes facebook/sam2-hiera-tiny from Hugging Face.
    """
    def __init__(self, device="cpu"):
        self.device = device
        logger.info("Initializing official SAM2 building segmentation model (facebook/sam2-hiera-tiny)...")
        try:
            self.processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-tiny")
            self.model = Sam2Model.from_pretrained("facebook/sam2-hiera-tiny").to(self.device)
            self.model.eval()
            logger.info("Official SAM2 loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load official SAM2 model: {e}")
            raise e

    def segment_buildings(self, image: Image.Image, bboxes: torch.Tensor) -> torch.Tensor:
        """
        Generates binary segmentation masks for detected buildings using SAM2.
        Args:
            image: input PIL image.
            bboxes: torch.Tensor of shape (N, 4) in coordinates [cx, cy, w, h] normalized to [0, 1].
        Returns:
            masks: torch.Tensor of shape (N, H, W) containing boolean binary masks.
        """
        num_boxes = len(bboxes)
        if num_boxes == 0:
            w_img, h_img = image.size
            return torch.zeros((0, h_img, w_img), dtype=torch.bool, device=self.device)
            
        logger.info("Executing official SAM2 segmentation inference...")
        w_img, h_img = image.size
        
        # Convert normalized [cx, cy, w, h] to absolute [x1, y1, x2, y2]
        input_boxes = []
        for box in bboxes:
            cx, cy, w, h = box[0].item(), box[1].item(), box[2].item(), box[3].item()
            x1 = max(0.0, (cx - w / 2.0) * w_img)
            y1 = max(0.0, (cy - h / 2.0) * h_img)
            x2 = min(float(w_img), (cx + w / 2.0) * w_img)
            y2 = min(float(h_img), (cy + h / 2.0) * h_img)
            input_boxes.append([x1, y1, x2, y2])
            
        # Format input boxes for processor: [batch_size, num_boxes_per_image, 4]
        input_boxes_t = torch.tensor([input_boxes], dtype=torch.float32).to(self.device)
        
        try:
            inputs = self.processor(images=image, input_boxes=input_boxes_t, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                outputs = self.model(**inputs)
                
            # Post-process masks
            masks = self.processor.post_process_masks(
                outputs.pred_masks, 
                inputs.original_sizes
            )[0]  # Get batch index 0
            
            # Select first multi-mask index, shape: (num_boxes, original_H, original_W)
            final_masks = masks[:, 0, :, :] > 0.0
            return final_masks
        except Exception as e:
            logger.error(f"Inference error during official SAM2 execution: {e}")
            raise e
