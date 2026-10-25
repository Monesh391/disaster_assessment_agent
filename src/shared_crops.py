"""
Shared Crop Extraction Module for xBD Disaster Damage Assessment
===============================================================
Ensures 100% parity between training, validation, evaluation, and production inference.

Guarantees:
1. Native 1024x1024 resolution cropping (never downsamples whole image first).
2. Exactly 1.3x contextual expansion around each building.
3. Square padding to strictly prevent aspect-ratio distortion.
4. Exact polygon rasterization producing a high-fidelity binary building mask (7th channel).
5. Robust boundary handling: out-of-bounds context is padded cleanly.
"""

import math
import numpy as np
import cv2
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from typing import Tuple, Optional, Union, List, Dict, Any
from shapely.geometry import Polygon


def extract_square_context_crop(
    pre_img: Union[np.ndarray, Image.Image],
    post_img: Union[np.ndarray, Image.Image],
    polygon_or_box: Union[Polygon, List[Tuple[float, float]], np.ndarray, List[float]],
    context_factor: float = 1.3,
    target_size: Tuple[int, int] = (128, 128),
    shift_offset: Tuple[float, float] = (0.0, 0.0),
    min_size_px: float = 16.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts a square context-padded crop and polygon mask from native 1024x1024 images.
    
    Args:
        pre_img: Pre-disaster image (H, W, 3) as numpy uint8 or PIL Image.
        post_img: Post-disaster image (H, W, 3) as numpy uint8 or PIL Image.
        polygon_or_box: Shapely Polygon, list of (x, y) vertices, or box [xmin, ymin, xmax, ymax].
        context_factor: Expansion factor around building (default 1.3x).
        target_size: Output (H, W) tuple (default 128x128).
        shift_offset: Normalized (dx, dy) in [-1.0, 1.0] for jitter augmentation.
        min_size_px: Minimum square crop size in native pixels.
        
    Returns:
        pre_crop: uint8 numpy array of shape (target_size[0], target_size[1], 3)
        post_crop: uint8 numpy array of shape (target_size[0], target_size[1], 3)
        mask_crop: uint8 numpy array of shape (target_size[0], target_size[1]) with 255 inside building
    """
    if isinstance(pre_img, Image.Image):
        pre_np = np.array(pre_img, dtype=np.uint8)
    else:
        pre_np = pre_img
        
    if isinstance(post_img, Image.Image):
        post_np = np.array(post_img, dtype=np.uint8)
    else:
        post_np = post_img
        
    H_img, W_img = pre_np.shape[:2]
    
    # 1. Extract vertices and bounding box in native coordinates
    poly_vertices = None
    if isinstance(polygon_or_box, Polygon):
        poly_vertices = np.array(polygon_or_box.exterior.coords, dtype=np.float32)
        xmin, ymin, xmax, ymax = polygon_or_box.bounds
    elif isinstance(polygon_or_box, (list, tuple, np.ndarray)):
        arr = np.array(polygon_or_box, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 2:
            poly_vertices = arr
            xmin = float(np.min(arr[:, 0]))
            ymin = float(np.min(arr[:, 1]))
            xmax = float(np.max(arr[:, 0]))
            ymax = float(np.max(arr[:, 1]))
        elif arr.ndim == 1 and len(arr) == 4:
            # Bounding box [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
            poly_vertices = np.array([
                [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]
            ], dtype=np.float32)
        else:
            raise ValueError(f"Unrecognized polygon_or_box format: {type(polygon_or_box)}, shape: {arr.shape}")
    else:
        raise TypeError(f"Unsupported geometry type: {type(polygon_or_box)}")
        
    bw = max(1.0, xmax - xmin)
    bh = max(1.0, ymax - ymin)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    
    # 2. Square dimension with context
    max_dim = max(bw, bh)
    crop_size = max(float(min_size_px), max_dim * context_factor)
    
    # Apply optional translation shift (augmentation)
    if shift_offset[0] != 0.0 or shift_offset[1] != 0.0:
        cx += shift_offset[0] * (crop_size * 0.10)
        cy += shift_offset[1] * (crop_size * 0.10)
        
    half_s = crop_size / 2.0
    x1 = cx - half_s
    y1 = cy - half_s
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    
    int_x1 = int(math.floor(x1))
    int_y1 = int(math.floor(y1))
    int_x2 = int(math.ceil(x2))
    int_y2 = int(math.ceil(y2))
    crop_w = max(1, int_x2 - int_x1)
    crop_h = max(1, int_y2 - int_y1)
    
    # Target square canvas size in native resolution
    canvas_s = max(crop_w, crop_h)
    
    # 3. Intersection with image boundaries
    src_x1 = max(0, min(W_img, int_x1))
    src_y1 = max(0, min(H_img, int_y1))
    src_x2 = max(0, min(W_img, int_x2))
    src_y2 = max(0, min(H_img, int_y2))
    
    # Canvas placement offsets
    dst_x1 = src_x1 - int_x1
    dst_y1 = src_y1 - int_y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    
    # 4. Create square canvases
    pre_canvas = np.zeros((canvas_s, canvas_s, 3), dtype=np.uint8)
    post_canvas = np.zeros((canvas_s, canvas_s, 3), dtype=np.uint8)
    mask_canvas = np.zeros((canvas_s, canvas_s), dtype=np.uint8)
    
    # Copy valid image regions into canvas
    if (src_x2 > src_x1) and (src_y2 > src_y1):
        pre_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = pre_np[src_y1:src_y2, src_x1:src_x2]
        post_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = post_np[src_y1:src_y2, src_x1:src_x2]
        
    # 5. Rasterize polygon mask on canvas
    if poly_vertices is not None and len(poly_vertices) >= 3:
        # Shift vertices relative to canvas origin
        shifted_verts = poly_vertices.copy()
        shifted_verts[:, 0] = shifted_verts[:, 0] - int_x1
        shifted_verts[:, 1] = shifted_verts[:, 1] - int_y1
        cv2.fillPoly(mask_canvas, [np.round(shifted_verts).astype(np.int32)], color=255)
    else:
        # Box fallback
        bx1 = int(round(xmin - int_x1))
        by1 = int(round(ymin - int_y1))
        bx2 = int(round(xmax - int_x1))
        by2 = int(round(ymax - int_y1))
        cv2.rectangle(mask_canvas, (bx1, by1), (bx2, by2), color=255, thickness=-1)
        
    # 6. Resize to target dimension (e.g. 128x128)
    out_w, out_h = target_size
    pre_resized = cv2.resize(pre_canvas, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    post_resized = cv2.resize(post_canvas, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask_canvas, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    
    return pre_resized, post_resized, mask_resized


def crops_to_7channel_tensor(
    pre_crop: np.ndarray,
    post_crop: np.ndarray,
    mask_crop: np.ndarray,
    normalize: bool = True
) -> torch.Tensor:
    """
    Converts pre, post, and mask crops into a 7-channel float tensor of shape (7, H, W).
    Channels 0..2: Pre RGB (normalized with ImageNet stats if normalize=True)
    Channels 3..5: Post RGB (normalized with ImageNet stats if normalize=True)
    Channel 6: Binary Building Mask in [0.0, 1.0]
    """
    pre_t = torch.from_numpy(pre_crop).permute(2, 0, 1).float() / 255.0
    post_t = torch.from_numpy(post_crop).permute(2, 0, 1).float() / 255.0
    mask_t = (torch.from_numpy(mask_crop).unsqueeze(0).float() / 255.0) > 0.5
    mask_t = mask_t.float()
    
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        pre_t = (pre_t - mean) / std
        post_t = (post_t - mean) / std
        
    return torch.cat([pre_t, post_t, mask_t], dim=0)  # Shape: (7, H, W)
