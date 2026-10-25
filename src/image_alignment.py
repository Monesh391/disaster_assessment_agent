"""
Satellite Image Alignment and Registration Module
=================================================
Provides robust geometric alignment between pre- and post-disaster satellite imagery.
Uses the before-disaster image as the spatial reference frame and transforms the
after-disaster image into its exact coordinate system.

Key Capabilities:
1. Multi-stage registration cascade: Feature-based (ORB + RANSAC) with fallback to
   intensity-based Enhanced Correlation Coefficient (ECC) registration.
2. Strict transformation matrix validation: Rejects singular, degenerate, sheared,
   extreme-scale, or extreme-rotation matrices.
3. Safe fallback: Gracefully falls back to dimension-resampled unwarped imagery when
   catastrophic disaster destruction or clouds prevent reliable registration.
4. Quality verification: Computes Normalized Cross-Correlation (NCC) to measure alignment
   quality and ensure warping did not degrade spatial coherence.
5. Visual verification overlays: Generates dual-temporal blend and difference maps.
"""

import math
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Union, Dict, Any

import cv2
import numpy as np
from PIL import Image


@dataclass
class AlignmentResult:
    """Detailed summary of the image registration process."""
    success: bool
    fallback_used: bool
    method: str
    warp_matrix: Optional[np.ndarray]
    dx: float
    dy: float
    rotation_deg: float
    scale: float
    inliers: int
    quality_score: float  # Normalized Cross-Correlation [-1.0, 1.0]
    status_message: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.warp_matrix is not None:
            d["warp_matrix"] = self.warp_matrix.tolist()
        return d


def validate_transformation_matrix(
    M: np.ndarray,
    img_w: int,
    img_h: int,
    max_translation_ratio: float = 0.25,
    min_scale: float = 0.80,
    max_scale: float = 1.25,
    max_rotation_deg: float = 25.0,
    max_shear: float = 0.20
) -> Tuple[bool, str]:
    """
    Validates whether an affine transformation matrix (2x3) represents a physically
    plausible satellite camera / georeferencing adjustment.
    
    Rejects:
    - Non-finite (NaN / Inf) values.
    - Negative or inverted determinants (reflections).
    - Excessive scale expansions/contractions outside [min_scale, max_scale].
    - Excessive rotational shifts > max_rotation_deg.
    - Excessive translations > max_translation_ratio * image dimension.
    - Excessive shearing (non-orthogonal transformations).
    """
    if M is None or not isinstance(M, np.ndarray):
        return False, "Matrix is None or not a numpy array"
    
    if M.shape != (2, 3):
        return False, f"Invalid matrix shape: {M.shape}, expected (2, 3)"
    
    if not np.all(np.isfinite(M)):
        return False, "Matrix contains NaN or Inf values"
    
    A = M[:, :2]
    t = M[:, 2]
    
    # Determinant = A00*A11 - A01*A10
    det = float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
    if det <= 0.0:
        return False, f"Non-positive determinant ({det:.4f}): indicates reflection or collapse"
    
    # Scale factor
    scale_x = math.hypot(float(A[0, 0]), float(A[1, 0]))
    scale_y = math.hypot(float(A[0, 1]), float(A[1, 1]))
    avg_scale = (scale_x + scale_y) / 2.0
    
    if avg_scale < min_scale or avg_scale > max_scale:
        return False, f"Scale factor ({avg_scale:.4f}) out of plausible range [{min_scale}, {max_scale}]"
    
    # Rotation angle (in degrees)
    rot_rad = math.atan2(float(A[1, 0]), float(A[0, 0]))
    rot_deg = math.degrees(rot_rad)
    if abs(rot_deg) > max_rotation_deg:
        return False, f"Rotation ({rot_deg:.2f} deg) exceeds threshold (+/- {max_rotation_deg} deg)"
    
    # Translation threshold
    dx, dy = float(t[0]), float(t[1])
    max_dx = img_w * max_translation_ratio
    max_dy = img_h * max_translation_ratio
    if abs(dx) > max_dx or abs(dy) > max_dy:
        return False, f"Translation (dx={dx:.1f}, dy={dy:.1f}) exceeds spatial bounds (max_dx={max_dx:.1f}, max_dy={max_dy:.1f})"
    
    # Orthogonality / Shear check: dot product of normalized basis vectors
    if scale_x > 1e-6 and scale_y > 1e-6:
        col0 = A[:, 0] / scale_x
        col1 = A[:, 1] / scale_y
        shear = abs(float(np.dot(col0, col1)))
        if shear > max_shear:
            return False, f"Affine shear ({shear:.4f}) exceeds threshold ({max_shear})"
            
    return True, "Valid transformation matrix"


def compute_normalized_cross_correlation(img1_gray: np.ndarray, img2_gray: np.ndarray) -> float:
    """
    Computes Normalized Cross-Correlation (NCC) between two grayscale images.
    Returns float in [-1.0, 1.0], where 1.0 indicates identical relative contrast/structures.
    """
    if img1_gray.shape != img2_gray.shape:
        return 0.0
    
    f1 = img1_gray.astype(np.float32)
    f2 = img2_gray.astype(np.float32)
    
    f1_mean = np.mean(f1)
    f2_mean = np.mean(f2)
    
    f1_zero = f1 - f1_mean
    f2_zero = f2 - f2_mean
    
    denom = math.sqrt(float(np.sum(f1_zero ** 2)) * float(np.sum(f2_zero ** 2)) + 1e-8)
    if denom < 1e-8:
        return 0.0
    
    ncc = float(np.sum(f1_zero * f2_zero) / denom)
    return max(-1.0, min(1.0, ncc))


def align_images_orb(
    img_ref_gray: np.ndarray,
    img_target_gray: np.ndarray,
    max_features: int = 3000,
    ratio_threshold: float = 0.80,
    min_inliers: int = 8
) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
    """
    Estimates an affine similarity transformation using ORB keypoint matching with RANSAC.
    """
    H, W = img_ref_gray.shape[:2]
    meta: Dict[str, Any] = {"inliers": 0, "total_matches": 0, "error": ""}
    
    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=15)
    kp_ref, des_ref = orb.detectAndCompute(img_ref_gray, None)
    kp_tgt, des_tgt = orb.detectAndCompute(img_target_gray, None)
    
    if des_ref is None or des_tgt is None or len(kp_ref) < 4 or len(kp_tgt) < 4:
        meta["error"] = "Insufficient keypoints detected"
        return False, None, meta
        
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        raw_matches = matcher.knnMatch(des_tgt, des_ref, k=2)
    except cv2.error as e:
        meta["error"] = f"KNN matching failed: {e}"
        return False, None, meta
        
    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
                
    meta["total_matches"] = len(good_matches)
    if len(good_matches) < min_inliers:
        meta["error"] = f"Too few good matches ({len(good_matches)} < {min_inliers})"
        return False, None, meta
        
    pts_tgt = np.float32([kp_tgt[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Estimate Euclidean/Similarity or Affine transformation: target -> ref
    M, inliers = cv2.estimateAffinePartial2D(
        pts_tgt,
        pts_ref,
        method=cv2.RANSAC,
        ransacReprojThreshold=4.0,
        maxIters=2000,
        confidence=0.99
    )
    
    inlier_count = int(np.sum(inliers)) if inliers is not None else 0
    meta["inliers"] = inlier_count
    
    if M is None or inlier_count < min_inliers:
        meta["error"] = f"RANSAC estimation failed or had insufficient inliers ({inlier_count} < {min_inliers})"
        return False, None, meta
        
    is_valid, reason = validate_transformation_matrix(M, W, H)
    if not is_valid:
        meta["error"] = f"Matrix rejected: {reason}"
        return False, None, meta
        
    return True, M, meta


def align_images_ecc(
    img_ref_gray: np.ndarray,
    img_target_gray: np.ndarray,
    motion_type: int = cv2.MOTION_EUCLIDEAN,
    max_iters: int = 60,
    eps: float = 1e-4,
    downsample_scale: float = 0.5
) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
    """
    Estimates a transformation using Enhanced Correlation Coefficient (ECC) registration.
    Runs on downsampled imagery for speed, robust basin of convergence, and noise immunity.
    """
    H, W = img_ref_gray.shape[:2]
    meta: Dict[str, Any] = {"correlation": 0.0, "error": ""}
    
    if downsample_scale < 1.0:
        sz = (int(W * downsample_scale), int(H * downsample_scale))
        small_ref = cv2.resize(img_ref_gray, sz, interpolation=cv2.INTER_AREA)
        small_tgt = cv2.resize(img_target_gray, sz, interpolation=cv2.INTER_AREA)
    else:
        small_ref = img_ref_gray
        small_tgt = img_target_gray
        
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iters, eps)
    
    try:
        cc, warp_matrix = cv2.findTransformECC(
            small_ref,
            small_tgt,
            warp_matrix,
            motion_type,
            criteria
        )
        meta["correlation"] = float(cc)
    except cv2.error as e:
        meta["error"] = f"ECC iteration failed to converge: {e}"
        return False, None, meta
    except Exception as e:
        meta["error"] = f"ECC failed: {e}"
        return False, None, meta
        
    # Scale translation back to full resolution if downsampled
    if downsample_scale < 1.0:
        warp_matrix[0, 2] /= downsample_scale
        warp_matrix[1, 2] /= downsample_scale
        
    is_valid, reason = validate_transformation_matrix(warp_matrix, W, H)
    if not is_valid:
        meta["error"] = f"ECC matrix rejected: {reason}"
        return False, None, meta
        
    return True, warp_matrix, meta


def align_satellite_pair(
    pre_img: Union[Image.Image, np.ndarray],
    post_img: Union[Image.Image, np.ndarray],
    prefer_method: str = "auto"
) -> Tuple[Union[Image.Image, np.ndarray], AlignmentResult]:
    """
    Performs high-fidelity satellite image alignment.
    
    Treats `pre_img` as the spatial reference frame and transforms `post_img`
    into `pre_img` coordinates.
    
    Args:
        pre_img: Pre-disaster reference image (PIL Image or numpy RGB uint8).
        post_img: Post-disaster target image (PIL Image or numpy RGB uint8).
        prefer_method: "auto", "orb", "ecc", or "none".
        
    Returns:
        aligned_post: The transformed post-disaster image in pre-disaster coordinates,
                      with the same type (PIL or numpy) and dimensions as pre_img.
        result: Detailed AlignmentResult metadata.
    """
    is_pil = isinstance(pre_img, Image.Image)
    
    if is_pil:
        pre_np = np.array(pre_img.convert("RGB"), dtype=np.uint8)
        post_np = np.array(post_img.convert("RGB"), dtype=np.uint8)
    else:
        pre_np = pre_img.copy()
        post_np = post_img.copy()
        
    ref_h, ref_w = pre_np.shape[:2]
    tgt_h, tgt_w = post_np.shape[:2]
    
    # Step 1: Dimension Normalization (if sizes differ)
    if (tgt_w, tgt_h) != (ref_w, ref_h):
        post_np_resized = cv2.resize(post_np, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
    else:
        post_np_resized = post_np.copy()
        
    # Convert to grayscale and apply contrast normalization (CLAHE) for feature robustness
    pre_gray = cv2.cvtColor(pre_np, cv2.COLOR_RGB2GRAY)
    post_gray = cv2.cvtColor(post_np_resized, cv2.COLOR_RGB2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    pre_gray_eq = clahe.apply(pre_gray)
    post_gray_eq = clahe.apply(post_gray)
    
    # Calculate baseline unaligned NCC
    baseline_ncc = compute_normalized_cross_correlation(pre_gray, post_gray)
    
    # Check if images are already perfectly identical
    if np.array_equal(pre_np, post_np_resized):
        identity_m = np.eye(2, 3, dtype=np.float32)
        res = AlignmentResult(
            success=True,
            fallback_used=False,
            method="Native-Identity",
            warp_matrix=identity_m,
            dx=0.0,
            dy=0.0,
            rotation_deg=0.0,
            scale=1.0,
            inliers=0,
            quality_score=1.0,
            status_message="Pre and post images are already identical in coordinates and content."
        )
        return (Image.fromarray(post_np_resized) if is_pil else post_np_resized), res

    chosen_M: Optional[np.ndarray] = None
    chosen_method: str = ""
    inliers_found: int = 0
    status_msg: str = ""
    
    # Step 2: Method selection & Execution
    if prefer_method in ["auto", "orb"]:
        ok, M_orb, meta_orb = align_images_orb(pre_gray_eq, post_gray_eq)
        if ok and M_orb is not None:
            chosen_M = M_orb
            chosen_method = "ORB-Affine"
            inliers_found = meta_orb.get("inliers", 0)
            status_msg = f"ORB feature alignment succeeded with {inliers_found} RANSAC inliers."
            
    if chosen_M is None and prefer_method in ["auto", "ecc"]:
        ok, M_ecc, meta_ecc = align_images_ecc(pre_gray_eq, post_gray_eq)
        if ok and M_ecc is not None:
            chosen_M = M_ecc
            chosen_method = "ECC-Euclidean"
            inliers_found = 0
            status_msg = f"ECC intensity alignment converged with correlation {meta_ecc.get('correlation', 0.0):.3f}."
            
    # Step 3: Validate and Apply Warp Matrix
    if chosen_M is not None:
        # Check matrix validity once more
        valid, val_reason = validate_transformation_matrix(chosen_M, ref_w, ref_h)
        if valid:
            # Warp post_np_resized into pre_np coordinates
            warped_post = cv2.warpAffine(
                post_np_resized,
                chosen_M,
                (ref_w, ref_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101
            )
            
            warped_gray = cv2.cvtColor(warped_post, cv2.COLOR_RGB2GRAY)
            warped_ncc = compute_normalized_cross_correlation(pre_gray, warped_gray)
            
            # Decompose transformation parameters
            A = chosen_M[:, :2]
            dx, dy = float(chosen_M[0, 2]), float(chosen_M[1, 2])
            scale_x = math.hypot(float(A[0, 0]), float(A[1, 0]))
            rot_deg = math.degrees(math.atan2(float(A[1, 0]), float(A[0, 0])))
            
            # If warping did not catastrophically ruin correlation:
            # (Allow modest decrease if disaster destroyed half the scene, but prevent random warp)
            if warped_ncc >= (baseline_ncc - 0.20):
                res = AlignmentResult(
                    success=True,
                    fallback_used=False,
                    method=chosen_method,
                    warp_matrix=chosen_M,
                    dx=dx,
                    dy=dy,
                    rotation_deg=rot_deg,
                    scale=scale_x,
                    inliers=inliers_found,
                    quality_score=warped_ncc,
                    status_message=f"{status_msg} Offset: dx={dx:+.1f}px, dy={dy:+.1f}px, rot={rot_deg:+.2f}°, scale={scale_x:.3f}."
                )
                return (Image.fromarray(warped_post) if is_pil else warped_post), res
            else:
                status_msg = f"Warping degraded correlation ({baseline_ncc:.3f} -> {warped_ncc:.3f}); triggering fallback."
        else:
            status_msg = f"Matrix validation failed: {val_reason}; triggering fallback."
            
    # Step 4: Safe Fallback
    res = AlignmentResult(
        success=False,
        fallback_used=True,
        method="Fallback-DimensionResampled",
        warp_matrix=None,
        dx=0.0,
        dy=0.0,
        rotation_deg=0.0,
        scale=1.0,
        inliers=0,
        quality_score=baseline_ncc,
        status_message=f"Alignment fell back to unwarped post-image. {status_msg}" if status_msg else "Feature matching & ECC correlation were insufficient; retained unwarped post-image."
    )
    return (Image.fromarray(post_np_resized) if is_pil else post_np_resized), res


def create_alignment_overlay(
    pre_img: Union[Image.Image, np.ndarray],
    aligned_post: Union[Image.Image, np.ndarray],
    mode: str = "blend",
    alpha: float = 0.50
) -> Image.Image:
    """
    Creates a visual verification overlay to visually inspect spatial registration.
    
    Args:
        pre_img: Reference image.
        aligned_post: Aligned post-disaster image.
        mode: "blend" (50/50 dual-temporal overlay), "difference" (absolute pixel diff),
              or "checkerboard" (alternating 64px spatial tiles).
        alpha: Transparency for blend mode.
        
    Returns:
        PIL Image of the overlay.
    """
    if isinstance(pre_img, Image.Image):
        p1 = pre_img.convert("RGB")
    else:
        p1 = Image.fromarray(pre_img).convert("RGB")
        
    if isinstance(aligned_post, Image.Image):
        p2 = aligned_post.convert("RGB")
    else:
        p2 = Image.fromarray(aligned_post).convert("RGB")
        
    if p1.size != p2.size:
        p2 = p2.resize(p1.size, Image.Resampling.BILINEAR)
        
    if mode == "blend":
        return Image.blend(p1, p2, alpha=alpha)
    elif mode == "difference":
        arr1 = np.array(p1, dtype=np.float32)
        arr2 = np.array(p2, dtype=np.float32)
        diff = np.abs(arr1 - arr2)
        # Enhance difference visibility
        diff_scaled = np.clip(diff * 2.0, 0, 255).astype(np.uint8)
        return Image.fromarray(diff_scaled)
    elif mode == "checkerboard":
        arr1 = np.array(p1, dtype=np.uint8)
        arr2 = np.array(p2, dtype=np.uint8)
        out = arr1.copy()
        tile_size = 64
        H, W = arr1.shape[:2]
        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                if ((x // tile_size) + (y // tile_size)) % 2 == 1:
                    y2 = min(H, y + tile_size)
                    x2 = min(W, x + tile_size)
                    out[y:y2, x:x2] = arr2[y:y2, x:x2]
        return Image.fromarray(out)
    else:
        return Image.blend(p1, p2, alpha=0.5)
