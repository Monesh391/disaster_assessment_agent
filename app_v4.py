import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from PIL import Image
import io
import hashlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, Point
import time
import os
import torch
import textwrap

# Import custom core modules
from src.config import config
from src.logger import logger
from src.spatial_grid import SpatialGridManager
from src.pdf_generator import generate_recovery_plan_pdf
from src.priority_ranking import PriorityRankingEngine
from src.agent import DisasterRecoveryAgent
from src.xbd_visualizer import DAMAGE_COLORS
from src.image_alignment import align_satellite_pair, create_alignment_overlay, AlignmentResult
from src.models.satellite_detector import SatelliteBuildingDetector
from src.models.sam2 import SAM2Model
from src.models.damage_net_v2 import SiameseDamageNetV2
from src.shared_crops import extract_square_context_crop, crops_to_7channel_tensor

# Select acceleration device and log strictly to terminal
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Hardware acceleration device selected: {device}")
print(f"[INFO] Hardware Acceleration Device Selected: {device.upper()}")

# Set page configuration to wide layout and professional title
st.set_page_config(
    page_title="Disaster Damage Assessment v2 (DamageNet v2 + YOLO 0.30)",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .title-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        color: white;
        padding: 2.2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
        border: 1px solid #475569;
    }
    
    .title-banner h1 {
        margin: 0;
        font-weight: 800;
        font-size: 2.2rem;
        letter-spacing: -0.5px;
        color: #f8fafc;
    }
    
    .title-banner p {
        margin: 0.6rem 0 0 0;
        font-weight: 400;
        font-size: 1.05rem;
        color: #cbd5e1;
        line-height: 1.5;
    }

    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    
    .kpi-title {
        font-size: 0.78rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.35rem;
    }
    
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0f172a;
    }

    .info-card {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Sleek tab navigation bar */
    div[data-testid="stRadio"] > div {
        display: flex;
        flex-direction: row;
        gap: 12px;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.6rem;
    }
    div[data-testid="stRadio"] label {
        background: #f8fafc;
        padding: 0.65rem 1.3rem;
        border-radius: 8px 8px 0 0;
        border: 1.5px solid #e2e8f0;
        border-bottom: none;
        cursor: pointer;
        font-weight: 600;
        color: #334155;
        transition: all 0.2s ease;
    }
    div[data-testid="stRadio"] label:hover {
        background: #e2e8f0;
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Application State Initialization
# ----------------------------------------------------
if "pipeline_executed" not in st.session_state:
    st.session_state.pipeline_executed = False
if "pre_image" not in st.session_state:
    st.session_state.pre_image = None
if "post_image" not in st.session_state:
    st.session_state.post_image = None
if "aligned_post_image" not in st.session_state:
    st.session_state.aligned_post_image = None
if "alignment_result" not in st.session_state:
    st.session_state.alignment_result = None
if "buildings" not in st.session_state:
    st.session_state.buildings = []
if "grid_ranked" not in st.session_state:
    st.session_state.grid_ranked = None
if "selected_cell_id" not in st.session_state:
    st.session_state.selected_cell_id = None
if "assessment_cache" not in st.session_state:
    st.session_state.assessment_cache = {}
if "active_section" not in st.session_state:
    st.session_state.active_section = "🏢 Building Assessment"
if "show_region_audit" not in st.session_state:
    st.session_state.show_region_audit = False
if "show_master_plan" not in st.session_state:
    st.session_state.show_master_plan = False
if "selected_audit_region" not in st.session_state:
    st.session_state.selected_audit_region = None
if "rendered_det_img" not in st.session_state:
    st.session_state.rendered_det_img = None
if "rendered_cls_img" not in st.session_state:
    st.session_state.rendered_cls_img = None
if "rendered_map_img" not in st.session_state:
    st.session_state.rendered_map_img = None
if "cached_master_plan_pdf" not in st.session_state:
    st.session_state.cached_master_plan_pdf = None

# Title Banner
st.markdown("""
<div class="title-banner">
    <h1>🛰️ Automated Disaster Damage Assessment & Spatial Prioritization</h1>
    <p>Upload pre- and post-disaster satellite imagery to automatically detect building footprints, assess structural damage levels, map high-risk spatial zones, and generate actionable recovery directives.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Visual Styling & Overlay Rendering Helpers (High-Resolution)
# ----------------------------------------------------
PALETTE_DAMAGE_COLORS = {
    "no-damage": "#10B981",       # 🟢 Emerald Green
    "minor-damage": "#eab308",    # 🟡 Amber Yellow
    "major-damage": "#f97316",    # 🟠 Vibrant Orange
    "destroyed": "#ef4444",       # 🔴 Crimson Red
    "un-classified": "#64748b"    # Slate Gray
}

def render_detection_overlay(pre_img, buildings) -> bytes:
    """Pre-renders ultra-sharp detection overlay into high-resolution PNG bytes."""
    fig, ax = plt.subplots(figsize=(10.24, 10.24), dpi=140)
    ax.imshow(pre_img)
    for b in buildings:
        poly = b.get("xy_geometry")
        if isinstance(poly, Polygon) and not poly.is_empty:
            x, y = poly.exterior.xy
            coords = list(zip(x, y))
            patch = patches.Polygon(coords, closed=True, edgecolor="#38bdf8", facecolor="#38bdf8", alpha=0.35, linewidth=1.5)
            ax.add_patch(patch)
            xmin, ymin, xmax, ymax = poly.bounds
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, edgecolor="#ef4444", facecolor="none", linewidth=1.1, linestyle="--")
            ax.add_patch(rect)
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.01, dpi=140)
    plt.close(fig)
    return buf.getvalue()

def render_damage_overlay(post_img, buildings) -> bytes:
    """Pre-renders ultra-sharp damage classification overlay into high-resolution PNG bytes."""
    fig, ax = plt.subplots(figsize=(10.24, 10.24), dpi=140)
    ax.imshow(post_img)
    for b in buildings:
        poly = b.get("xy_geometry")
        damage = b.get("damage_label", "no-damage")
        color = PALETTE_DAMAGE_COLORS.get(damage, "#10B981")
        if isinstance(poly, Polygon) and not poly.is_empty:
            x, y = poly.exterior.xy
            coords = list(zip(x, y))
            patch = patches.Polygon(coords, closed=True, edgecolor=color, facecolor=color, alpha=0.62, linewidth=1.8)
            ax.add_patch(patch)
    ax.axis("off")
    legend_patches = [
        patches.Patch(color=PALETTE_DAMAGE_COLORS["no-damage"], label="🟢 No Damage"),
        patches.Patch(color=PALETTE_DAMAGE_COLORS["minor-damage"], label="🟡 Minor Damage"),
        patches.Patch(color=PALETTE_DAMAGE_COLORS["major-damage"], label="🟠 Major Damage"),
        patches.Patch(color=PALETTE_DAMAGE_COLORS["destroyed"], label="🔴 Destroyed")
    ]
    ax.legend(handles=legend_patches, loc="upper right", framealpha=0.92, fontsize=10.0)
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.01, dpi=140)
    plt.close(fig)
    return buf.getvalue()

def render_spatial_priority_map(post_img, grid_ranked, buildings) -> bytes:
    """Pre-renders ultra-sharp 2x2 spatial priority map overlay with clear dual legends into PNG bytes."""
    fig, ax = plt.subplots(figsize=(10.24, 10.24), dpi=140)
    ax.imshow(post_img)
    
    # 1. Shaded quadrant fill and bold border
    for idx, row in grid_ranked.iterrows():
        tier_label = str(row.get("priority_label", "Recover Later"))
        color = str(row.get("tier_color", "#22c55e"))
        alpha = 0.28 if color == "#ef4444" else (0.20 if color == "#f97316" else (0.14 if color == "#eab308" else 0.08))
        
        x_min, y_min = row["x_min"], row["y_min"]
        x_max, y_max = row["x_max"], row["y_max"]
        w = x_max - x_min
        h = y_max - y_min
        
        rect_fill = patches.Rectangle((x_min, y_min), w, h, linewidth=0, facecolor=color, alpha=alpha)
        ax.add_patch(rect_fill)
        
        lw = 3.6 if color == "#ef4444" else (2.6 if color == "#f97316" else 1.8)
        rect_border = patches.Rectangle((x_min, y_min), w, h, linewidth=lw, edgecolor=color, facecolor="none")
        ax.add_patch(rect_border)
        
        badge_text = str(row['grid_id']).upper() + '\n' + str(tier_label).upper()
        ax.text(
            x_min + 22, y_min + 35, badge_text,
            fontsize=10.5, color="white", weight="bold", linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#0f172a", edgecolor=color, alpha=0.94, linewidth=2.0)
        )
        
    # 2. Draw classified building polygons
    for b in buildings:
        poly = b.get("xy_geometry")
        if isinstance(poly, Polygon) and not poly.is_empty:
            dmg = b.get("damage_label", "no-damage")
            b_color = PALETTE_DAMAGE_COLORS.get(dmg, "#10B981")
            x, y = poly.exterior.xy
            patch = patches.Polygon(list(zip(x, y)), closed=True, edgecolor=b_color, facecolor=b_color, alpha=0.70, linewidth=1.6)
            ax.add_patch(patch)
            
    # 3. Two-part clean legends: Recovery Priority Ranks AND Building Damage Classifications
    area_legend = [
        patches.Patch(color="#ef4444", label="🔴 Rank 1 — Recover First"),
        patches.Patch(color="#f97316", label="🟠 Rank 2 — Recover Next"),
        patches.Patch(color="#eab308", label="🟡 Rank 3 — Moderate Priority"),
        patches.Patch(color="#22c55e", label="🟢 Rank 4 — Recover Later")
    ]
    damage_legend = [
        patches.Patch(color=PALETTE_DAMAGE_COLORS["no-damage"], label="🟢 No Damage"),
        patches.Patch(color=PALETTE_DAMAGE_COLORS["minor-damage"], label="🟡 Minor Damage"),
        patches.Patch(color=PALETTE_DAMAGE_COLORS["major-damage"], label="🟠 Major Damage"),
        patches.Patch(color=PALETTE_DAMAGE_COLORS["destroyed"], label="🔴 Destroyed")
    ]
    leg1 = ax.legend(handles=area_legend, title="Spatial Priority", loc="lower right", framealpha=0.92, fontsize=9.0, title_fontsize=10.0)
    ax.add_artist(leg1)
    ax.legend(handles=damage_legend, title="Building Damage", loc="upper right", framealpha=0.92, fontsize=9.0, title_fontsize=10.0)
    
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.01, dpi=140)
    plt.close(fig)
    return buf.getvalue()

def render_html_block(html_content: str):
    """Renders pure HTML without allowing CommonMark to treat indented tags as code blocks."""
    cleaned = "\n".join(line.strip() for line in html_content.strip().splitlines() if line.strip())
    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_models_cached():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        num_cores = os.cpu_count() or 4
        torch.set_num_threads(min(8, num_cores))
    detector = SatelliteBuildingDetector(model_path="models/satellite_yolo_detector.pt", device=device)
    # SAM2 inference bypassed: downstream DamageNet v2 classifier & UI overlays consume rectangular context crops
    sam2 = None
    damage_net = SiameseDamageNetV2(backbone_name="resnet18", pretrained=False, in_channels_per_branch=4, num_classes=4).to(device)
    ckpt_path = "models/damage_net_v2_step4_resnet18_ce.pth"
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                damage_net.load_state_dict(ckpt["model_state_dict"])
            elif isinstance(ckpt, dict):
                damage_net.load_state_dict(ckpt)
            logger.info(f"Loaded Siamese DamageNet v2 weights from {ckpt_path} successfully.")
        except Exception as e:
            logger.error(f"Error loading Siamese DamageNet v2 weights: {e}")
    damage_net.eval()
    return detector, sam2, damage_net

@st.cache_resource(show_spinner=False)
def load_agent_cached():
    return DisasterRecoveryAgent()

# Pre-warm cached models and agent on application startup
detector_model, sam2_model, damage_model = load_models_cached()
agent = load_agent_cached()

# ----------------------------------------------------
# Sidebar Inputs
# ----------------------------------------------------
st.sidebar.header("📥 Satellite Imagery Input")
st.sidebar.markdown("Upload before and after satellite images to perform automated damage assessment.")

pre_file = st.sidebar.file_uploader(
    "Before Disaster Satellite Image", 
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    help="Upload pre-disaster high-resolution satellite imagery."
)

post_file = st.sidebar.file_uploader(
    "After Disaster Satellite Image", 
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    help="Upload post-disaster high-resolution satellite imagery."
)

st.sidebar.markdown("---")
execute_pipeline = st.sidebar.button("Run Damage Assessment", type="primary", use_container_width=True)

# ----------------------------------------------------
# Pipeline Processing Logic
# ----------------------------------------------------
if execute_pipeline:
    if pre_file is None or post_file is None:
        st.sidebar.error("⚠️ Please upload both Before and After disaster satellite images to proceed.")
    else:
        try:
            pre_bytes = pre_file.getvalue()
            post_bytes = post_file.getvalue()
            cache_key = hashlib.sha256(pre_bytes + b"::" + post_bytes).hexdigest()
            
            # Check if this exact image pair has already been computed
            if cache_key in st.session_state.assessment_cache:
                cached_res = st.session_state.assessment_cache[cache_key]
                st.session_state.pre_image = cached_res["pre_image"]
                st.session_state.post_image = cached_res["post_image"]
                st.session_state.aligned_post_image = cached_res["aligned_post_image"]
                st.session_state.alignment_result = cached_res.get("alignment_result", None)
                st.session_state.buildings = cached_res["buildings"]
                
                # Always recalculate fresh spatial grid ranking with current priority logic
                pre_w, pre_h = st.session_state.pre_image.size
                grid_mgr = SpatialGridManager(
                    image_width=pre_w,
                    image_height=pre_h,
                    grid_cols=2,
                    grid_rows=2,
                    mode="image_space"
                )
                grid_df = grid_mgr.generate_grid()
                st.session_state.grid_ranked = grid_mgr.assign_and_prioritize(grid_df, st.session_state.buildings)
                st.session_state.pipeline_executed = True
                st.session_state["cached_master_plan_pdf"] = None
                
                # Check if rendered visuals are in cache, otherwise generate them
                if "rendered_det_img" in cached_res and cached_res["rendered_det_img"] is not None:
                    st.session_state.rendered_det_img = cached_res["rendered_det_img"]
                    st.session_state.rendered_cls_img = cached_res["rendered_cls_img"]
                    st.session_state.rendered_map_img = cached_res["rendered_map_img"]
                else:
                    st.session_state.rendered_det_img = render_detection_overlay(st.session_state.pre_image, st.session_state.buildings)
                    st.session_state.rendered_cls_img = render_damage_overlay(st.session_state.aligned_post_image, st.session_state.buildings)
                    st.session_state.rendered_map_img = render_spatial_priority_map(st.session_state.aligned_post_image, st.session_state.grid_ranked, st.session_state.buildings)
                
                # Check if rendered visuals are in cache, otherwise generate them
                if "rendered_det_img" in cached_res and cached_res["rendered_det_img"] is not None:
                    st.session_state.rendered_det_img = cached_res["rendered_det_img"]
                    st.session_state.rendered_cls_img = cached_res["rendered_cls_img"]
                    st.session_state.rendered_map_img = cached_res["rendered_map_img"]
                else:
                    st.session_state.rendered_det_img = render_detection_overlay(st.session_state.pre_image, st.session_state.buildings)
                    st.session_state.rendered_cls_img = render_damage_overlay(st.session_state.aligned_post_image, st.session_state.buildings)
                    st.session_state.rendered_map_img = render_spatial_priority_map(st.session_state.aligned_post_image, st.session_state.grid_ranked, st.session_state.buildings)
                
                print("\n" + "="*55)
                print(">>> DAMAGE ASSESSMENT PIPELINE EXECUTION (Terminal Only)")
                print(f"  * Device:               {device.upper()} (Cached Result Reused)")
                print("  * Cache Status:         CACHE HIT (0.0000 s)")
                print("  * TOTAL PIPELINE TIME:  0.0001 s")
                print("="*55 + "\n")
            else:
                progress_placeholder = st.empty()
                with progress_placeholder.container():
                    with st.status("Analyzing satellite imagery...", expanded=True) as status_box:
                        t_total_start = time.time()
                        
                        # 1. Image Preprocessing & Spatial Dimension Alignment
                        t_prep_start = time.time()
                        pre_image = Image.open(io.BytesIO(pre_bytes)).convert("RGB")
                        post_image = Image.open(io.BytesIO(post_bytes)).convert("RGB")
                        
                        st.write("Aligning dual-temporal satellite imagery...")
                        pre_w, pre_h = pre_image.size
                        aligned_post_image, align_res = align_satellite_pair(pre_image, post_image, prefer_method="auto")
                        t_prep = time.time() - t_prep_start
                        
                        # 2. Satellite YOLO Building Localization
                        st.write("Detecting buildings...")
                        t_det_start = time.time()
                        bboxes = detector_model.detect_buildings(
                            pre_image,
                            conf_threshold=0.30,
                            iou_threshold=0.50,
                            imgsz=1024
                        )
                        t_det = time.time() - t_det_start
                        
                        # 3. Structural Damage Classification (SAM2 bypassed: downstream classifier consumes calibrated rectangular context crops)
                        st.write("Assessing structural damage...")
                        t_sam2 = 0.0  # SAM2 bypassed: eliminates redundant segmentation inference
                        
                        t_vit_start = time.time()
                        damage_labels = ["no-damage", "minor-damage", "major-damage", "destroyed"]
                        crop_tensors = []
                        crop_metadata = []
                        
                        for idx, box in enumerate(bboxes):
                            cx, cy, w, h = box[0].item(), box[1].item(), box[2].item(), box[3].item()

                            cx_px, cy_px = cx * pre_w, cy * pre_h
                            bw_px, bh_px = w * pre_w, h * pre_h

                            x1 = int(max(0, cx_px - bw_px / 2.0))
                            y1 = int(max(0, cy_px - bh_px / 2.0))
                            x2 = int(min(pre_w, cx_px + bw_px / 2.0))
                            y2 = int(min(pre_h, cy_px + bh_px / 2.0))
                            
                            box_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

                            pre_c, post_c, mask_c = extract_square_context_crop(
                                pre_img=pre_image,
                                post_img=aligned_post_image,
                                polygon_or_box=box_poly,
                                context_factor=1.3,
                                target_size=(128, 128)
                            )
                            tensor_7ch = crops_to_7channel_tensor(pre_c, post_c, mask_c)
                            crop_tensors.append(tensor_7ch)
                            crop_metadata.append((idx, cx, cy, w, h, x1, y1, x2, y2))
                            
                        if len(crop_tensors) > 0:
                            damage_model.eval()
                            batch_tensors = torch.stack(crop_tensors).to(device)
                            with torch.no_grad():
                                logits = damage_model(batch_tensors)
                                # Validation-calibrated class scaling multipliers to optimize Minor & Major F1
                                calib_multipliers = torch.tensor([0.85, 1.10, 0.80, 1.35], dtype=torch.float32, device=device)
                                calibrated_probs = torch.softmax(logits, dim=1) * calib_multipliers
                                pred_classes = torch.argmax(calibrated_probs, dim=1).cpu().tolist()
                        else:
                            pred_classes = []
                            
                        buildings_raw = []
                        for (idx, cx, cy, w, h, x1, y1, x2, y2), pred_class in zip(crop_metadata, pred_classes):
                            lbl = damage_labels[pred_class]
                            polygon_xy = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
                            
                            buildings_raw.append({
                                "uid": f"BLD_{idx:04d}",
                                "xy_geometry": polygon_xy,
                                "damage_label": lbl,
                                "damage_class": pred_class
                            })
                        t_vit = time.time() - t_vit_start
                        
                        # 4. Spatial Grid Dynamic Prioritization
                        st.write("Calculating recovery priorities...")
                        t_grid_start = time.time()
                        grid_mgr = SpatialGridManager(
                            image_width=pre_w,
                            image_height=pre_h,
                            grid_cols=2,
                            grid_rows=2,
                            mode="image_space"
                        )
                        grid_df = grid_mgr.generate_grid()
                        grid_ranked = grid_mgr.assign_and_prioritize(grid_df, buildings_raw)
                        t_grid = time.time() - t_grid_start
                        
                        # 5. AI Recovery Agent Plan Compilation
                        t_agent_start = time.time()
                        d_counts = {"no-damage": 0, "minor-damage": 0, "major-damage": 0, "destroyed": 0}
                        for b in buildings_raw:
                            lbl = b["damage_label"]
                            if lbl in d_counts:
                                d_counts[lbl] += 1
                        _ = agent.generate_recovery_plan(grid_ranked, d_counts)
                        t_agent = time.time() - t_agent_start
                        
                        t_total = t_prep + t_det + t_sam2 + t_vit + t_grid + t_agent
                        status_box.update(label="Analyzing satellite imagery...", state="complete", expanded=False)
                
                # Clear progress container immediately upon completion
                progress_placeholder.empty()
                
                # Terminal-only timing output (never rendered in Streamlit UI)
                print("\n" + "="*55)
                print(">>> DAMAGE ASSESSMENT PIPELINE EXECUTION TIMING (Terminal Only)")
                print(f"  * Device:               {device.upper()}")
                print(f"  * Image Preprocessing:  {t_prep:.4f} s")
                print(f"  * YOLO Detection:       {t_det:.4f} s")
                print(f"  * SAM2 Segmentation:    BYPASSED (0.0000 s - Redundant inference disabled)")
                print(f"  * ViT Classification:   {t_vit:.4f} s")
                print(f"  * Spatial Grid:         {t_grid:.4f} s")
                print(f"  * AI Recommendation:    {t_agent:.4f} s")
                print(f"  * TOTAL PIPELINE TIME:  {t_total:.4f} s")
                print("="*55 + "\n")
                
                # Pre-render visuals immediately for instant UI navigation
                st.session_state.rendered_det_img = render_detection_overlay(pre_image, buildings_raw)
                st.session_state.rendered_cls_img = render_damage_overlay(aligned_post_image, buildings_raw)
                st.session_state.rendered_map_img = render_spatial_priority_map(aligned_post_image, grid_ranked, buildings_raw)

                # Update Session State
                st.session_state.pre_image = pre_image
                st.session_state.post_image = post_image
                st.session_state.aligned_post_image = aligned_post_image
                st.session_state.alignment_result = align_res
                st.session_state.buildings = buildings_raw
                st.session_state.grid_ranked = grid_ranked
                st.session_state.pipeline_executed = True
                st.session_state["cached_master_plan_pdf"] = None
                
                # Save to cache
                st.session_state.assessment_cache[cache_key] = {
                    "pre_image": pre_image,
                    "post_image": post_image,
                    "aligned_post_image": aligned_post_image,
                    "alignment_result": align_res,
                    "buildings": buildings_raw,
                    "grid_ranked": grid_ranked,
                    "rendered_det_img": st.session_state.rendered_det_img,
                    "rendered_cls_img": st.session_state.rendered_cls_img,
                    "rendered_map_img": st.session_state.rendered_map_img
                }
                
        except Exception as e:
            logger.error(f"Error during assessment execution: {e}")
            st.error(f"An error occurred while processing the satellite imagery: {e}")

# ----------------------------------------------------
# Main Dashboard UI Layout
# ----------------------------------------------------
if not st.session_state.pipeline_executed:
    st.info("👈 Please upload both Before and After disaster satellite images in the sidebar and click **Run Damage Assessment**.")
else:
    # Damage counts
    # Robust damage counts derived from grid_ranked to ensure numbers never disappear
    if st.session_state.grid_ranked is not None and len(st.session_state.grid_ranked) > 0:
        total_blds = int(st.session_state.grid_ranked["total_buildings"].sum())
        damage_counts = {
            "no-damage": int(st.session_state.grid_ranked["count_no_damage"].sum()),
            "minor-damage": int(st.session_state.grid_ranked["count_minor_damage"].sum()),
            "major-damage": int(st.session_state.grid_ranked["count_major_damage"].sum()),
            "destroyed": int(st.session_state.grid_ranked["count_destroyed"].sum())
        }
    else:
        total_blds = len(st.session_state.buildings)
        damage_counts = {"no-damage": 0, "minor-damage": 0, "major-damage": 0, "destroyed": 0}
        for b in st.session_state.buildings:
            lbl = b.get("damage_label", "no-damage")
            if lbl in damage_counts:
                damage_counts[lbl] += 1
                
    damaged_total = damage_counts["minor-damage"] + damage_counts["major-damage"] + damage_counts["destroyed"]
    critical_sectors_count = len(st.session_state.grid_ranked[st.session_state.grid_ranked["priority_tier"] == "CRITICAL"])
    high_sectors_count = len(st.session_state.grid_ranked[st.session_state.grid_ranked["priority_tier"] == "HIGH"])
    
    rank_to_tier = {
        1: ("RECOVER FIRST", "#ef4444", "🔴", 0.28),
        2: ("RECOVER NEXT", "#f97316", "🟠", 0.20),
        3: ("MODERATE PRIORITY", "#eab308", "🟡", 0.14),
        4: ("RECOVER LATER", "#22c55e", "🟢", 0.08)
    }
    rank_to_tier_simple = {
        1: ("Recover First", "#ef4444", "🔴"),
        2: ("Recover Next", "#f97316", "🟠"),
        3: ("Moderate Priority", "#eab308", "🟡"),
        4: ("Recover Later", "#22c55e", "🟢")
    }
    
    SECTIONS = [
        "🏢 Building Assessment",
        "🗺️ Spatial Priority Assessment",
        "🚨 Recovery Recommendation"
    ]
    
    if "active_section" not in st.session_state or st.session_state["active_section"] not in SECTIONS:
        st.session_state["active_section"] = SECTIONS[0]
        
    active_section = st.radio(
        "Navigation",
        SECTIONS,
        horizontal=True,
        key="active_section",
        label_visibility="collapsed"
    )

    # ====================================================
    # SECTION 1: BUILDING ASSESSMENT
    # ====================================================
    if active_section == "🏢 Building Assessment":
        st.subheader("🏢 Building Footprints & Structural Damage Assessment")
        st.markdown("Detailed breakdown of detected building footprints and their evaluated structural damage classifications.")
        
        pct_no_dmg = (damage_counts["no-damage"] / max(1, total_blds)) * 100
        pct_minor = (damage_counts["minor-damage"] / max(1, total_blds)) * 100
        pct_major = (damage_counts["major-damage"] / max(1, total_blds)) * 100
        pct_destroyed = (damage_counts["destroyed"] / max(1, total_blds)) * 100
        
        col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
        with col_b1:
            st.metric("Total Detected", total_blds)
        with col_b2:
            st.metric("🟢 No Damage", f"{damage_counts['no-damage']}", f"{pct_no_dmg:.1f}%")
        with col_b3:
            st.metric("🟡 Minor Damage", f"{damage_counts['minor-damage']}", f"{pct_minor:.1f}%")
        with col_b4:
            st.metric("🟠 Major Damage", f"{damage_counts['major-damage']}", f"{pct_major:.1f}%")
        with col_b5:
            st.metric("🔴 Destroyed", f"{damage_counts['destroyed']}", f"{pct_destroyed:.1f}%")

        # Dual-Temporal Image Alignment Status & Visual Verification Overlay
        if st.session_state.get("alignment_result") is not None:
            a_res = st.session_state.alignment_result
            status_color = "#16a34a" if (a_res.success and not a_res.fallback_used) else "#ca8a04"
            status_icon = "✅" if (a_res.success and not a_res.fallback_used) else "⚠️"
            
            with st.expander("🛰️ Temporal Image Alignment & Geometric Registration Status", expanded=False):
                align_summary_html = f"""
<div style="background: #f8fafc; border: 1px solid #cbd5e1; border-left: 5px solid {status_color}; border-radius: 8px; padding: 0.85rem 1.1rem; margin-bottom: 0.8rem;">
    <div style="font-weight: 700; font-size: 0.98rem; color: #0f172a; margin-bottom: 0.25rem;">
        {status_icon} Registration Method: <b>{a_res.method}</b> &nbsp;|&nbsp; Status: <b>{'SUCCESS' if a_res.success else 'FALLBACK ACTIVE'}</b>
    </div>
    <div style="font-size: 0.86rem; color: #334155; line-height: 1.45;">
        {a_res.status_message}
    </div>
</div>
"""
                render_html_block(align_summary_html)
                
                col_a1, col_a2, col_a3, col_a4 = st.columns(4)
                with col_a1:
                    st.metric("Shift X (dx)", f"{a_res.dx:+.1f} px")
                with col_a2:
                    st.metric("Shift Y (dy)", f"{a_res.dy:+.1f} px")
                with col_a3:
                    st.metric("Rotation", f"{a_res.rotation_deg:+.2f}°")
                with col_a4:
                    st.metric("Correlation (NCC)", f"{a_res.quality_score:.3f}")
                    
                overlay_mode = st.radio(
                    "Select Verification Overlay Mode",
                    ["50/50 Dual-Temporal Blend", "Checkerboard Grid", "Pixel Difference Map"],
                    horizontal=True,
                    key="overlay_verification_mode"
                )
                mode_key = "blend" if "Blend" in overlay_mode else ("checkerboard" if "Checkerboard" in overlay_mode else "difference")
                overlay_img = create_alignment_overlay(st.session_state.pre_image, st.session_state.aligned_post_image, mode=mode_key)
                st.image(
                    overlay_img,
                    caption=f"Dual-Temporal Alignment Verification ({overlay_mode}) — Verifies spatial coincidence of roads and landmark boundaries.",
                    use_container_width=True
                )

        st.markdown("---")
        
        col_v1, col_v2 = st.columns(2)
        
        with col_v1:
            st.markdown("#### 📦 Pre-Disaster: Building Footprint Detections")
            if st.session_state.rendered_det_img is None:
                st.session_state.rendered_det_img = render_detection_overlay(st.session_state.pre_image, st.session_state.buildings)
            st.image(st.session_state.rendered_det_img, use_container_width=True)
            
        with col_v2:
            st.markdown("#### 🎨 Post-Disaster: Classified Structural Damage")
            if st.session_state.rendered_cls_img is None:
                st.session_state.rendered_cls_img = render_damage_overlay(st.session_state.aligned_post_image, st.session_state.buildings)
            st.image(st.session_state.rendered_cls_img, use_container_width=True)

    # ====================================================
    # SECTION 2: SPATIAL RECOVERY PRIORITY
    # ====================================================
    elif active_section == "🗺️ Spatial Priority Assessment":
        st.subheader("🗺️ Spatial Recovery Priority")
        st.markdown(
            "Regions with higher damage priority should be assessed and recovered first. The after-disaster satellite imagery is partitioned into four dynamic quadrants."
        )
        
        # 1. Damage Assessment Summary Bar
        st.markdown("### 📊 Damage Assessment Summary")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            st.metric("🟢 No Damage", f"{damage_counts['no-damage']}")
        with col_d2:
            st.metric("🟡 Minor Damage", f"{damage_counts['minor-damage']}")
        with col_d3:
            st.metric("🟠 Major Damage", f"{damage_counts['major-damage']}")
        with col_d4:
            st.metric("🔴 Destroyed", f"{damage_counts['destroyed']}")
            
        st.markdown("---")
        
        # 2. Prominent HIGHEST RECOVERY PRIORITY Banner
        top_area = st.session_state.grid_ranked.iloc[0]
        top_area_name = str(top_area["grid_id"]).upper()
        top_des = int(top_area.get("count_destroyed", 0))
        top_maj = int(top_area.get("count_major_damage", 0))
        top_min = int(top_area.get("count_minor_damage", 0))
        has_active_damage = (top_des > 0) or (top_maj > 0) or (top_min > 0)
        
        if has_active_damage:
            top_banner_html = textwrap.dedent(f"""
            <div style="background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%); color: white; padding: 1.4rem 1.8rem; border-radius: 12px; border: 1.5px solid #ef4444; margin-bottom: 1.3rem; box-shadow: 0 4px 14px rgba(239, 68, 68, 0.25);">
                <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #fca5a5; margin-bottom: 0.25rem;">
                    🚨 HIGHEST RECOVERY PRIORITY — RECOVER FIRST
                </div>
                <h2 style="color: white; margin: 0; font-size: 1.85rem; font-weight: 800; letter-spacing: -0.5px;">
                    {top_area_name} &nbsp;—&nbsp; RECOVER FIRST
                </h2>
                <p style="color: #fee2e2; margin: 0.5rem 0 0 0; font-size: 1.0rem; font-weight: 500; line-height: 1.4;">
                    Recovery prioritization is strictly determined by building damages. <b>{top_area_name}</b> has the highest detected damage severity ({top_des} destroyed, {top_maj} major damage, {top_min} minor damage) and is assigned primary emergency response.
                </p>
            </div>
            """).strip()
        else:
            top_banner_html = textwrap.dedent(f"""
            <div style="background: linear-gradient(135deg, #14532d 0%, #166534 100%); color: white; padding: 1.4rem 1.8rem; border-radius: 12px; border: 1.5px solid #22c55e; margin-bottom: 1.3rem; box-shadow: 0 4px 14px rgba(34, 197, 94, 0.2);">
                <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #86efac; margin-bottom: 0.25rem;">
                    🟢 BASELINE MONITORING STATUS
                </div>
                <h2 style="color: white; margin: 0; font-size: 1.85rem; font-weight: 800; letter-spacing: -0.5px;">
                    NO CRITICAL STRUCTURAL DAMAGE DETECTED
                </h2>
                <p style="color: #dcfce7; margin: 0.5rem 0 0 0; font-size: 1.0rem; font-weight: 500; line-height: 1.4;">
                    No destroyed or major structural damage was identified across surveyed sectors. All sectors are assigned routine municipal monitoring status.
                </p>
            </div>
            """).strip()
        st.html(top_banner_html)
        
        # 3. Dynamic 2x2 Spatial Priority Map on After-Disaster Satellite Image
        if st.session_state.rendered_map_img is None:
            st.session_state.rendered_map_img = render_spatial_priority_map(st.session_state.aligned_post_image, st.session_state.grid_ranked, st.session_state.buildings)
        st.image(st.session_state.rendered_map_img, use_container_width=True)

    # SECTION 3: RECOVERY RECOMMENDATION
    # ====================================================
    elif active_section == "🚨 Recovery Recommendation":
        st.subheader("🚨 Emergency Recovery Directives & Recommendations")
        st.markdown(
            "Automated recovery planning and response prioritization based on spatial damage concentration and risk scoring."
        )
        
        # 1. Ranked Recovery Priority List
        st.markdown("### 📋 Recovery Priority")
        st.markdown("*The region where the damage occurs more is prioritized to be recovered first. Use the Operational Directives below to inspect specific regions.*")
        
        for idx, row in st.session_state.grid_ranked.iterrows():
            tier_label = str(row.get("priority_label", "Recover Later"))
            tier_color = str(row.get("tier_color", "#22c55e"))
            tier_icon = str(row.get("tier_icon", "🟢"))
            rank_str = str(row.get("priority_rank", f"Sector #{idx + 1}"))
            
            area_name = row["grid_id"]
            b_total = int(row["total_buildings"])
            b_maj = int(row["count_major_damage"])
            b_des = int(row["count_destroyed"])
            b_min = int(row["count_minor_damage"])
            b_nodmg = int(row["count_no_damage"])
            
            p_card_html = f"""<div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 6px solid {tier_color}; border-radius: 8px; padding: 0.85rem 1.2rem; margin-bottom: 0.4rem; box-shadow: 0 1px 3px rgba(0,0,0,0.03); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;"><div style="font-size: 1.05rem; font-weight: 700; color: #0f172a;">{tier_icon} <b>{area_name}</b> — <span style="color: {tier_color};">{tier_label} ({rank_str})</span></div><div style="font-size: 0.88rem; color: #334155; font-weight: 500;"><b>{b_total}</b> buildings &nbsp;|&nbsp; <span style="color: #ef4444; font-weight: 600;">{b_des} destroyed</span> &nbsp;|&nbsp; <span style="color: #f97316; font-weight: 600;">{b_maj} major</span> &nbsp;|&nbsp; <span style="color: #ca8a04; font-weight: 600;">{b_min} minor</span> &nbsp;|&nbsp; <span style="color: #16a34a; font-weight: 600;">{b_nodmg} undamaged</span></div></div>"""
            
            render_html_block(p_card_html)

        # 2. Sector Priority Breakdown Table
        st.markdown("#### 📋 Sector Priority & Action Breakdown")
        table_rows = []
        for idx, row in st.session_state.grid_ranked.iterrows():
            tier_label = str(row.get("priority_label", "Recover Later"))
            table_rows.append({
                "Rank": str(row.get("priority_rank", f"#{idx + 1}")),
                "Region": row["grid_id"],
                "Priority Level": tier_label,
                "Buildings Assessed": int(row["total_buildings"]),
                "Destroyed": int(row["count_destroyed"]),
                "Major Damage": int(row["count_major_damage"]),
                "Minor Damage": int(row["count_minor_damage"]),
                "Undamaged": int(row["count_no_damage"])
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # 3. Operational Controls
        st.markdown("### 📋 Operational Directives")
        
        if st.session_state.grid_ranked is not None and len(st.session_state.grid_ranked) > 0:
            all_grid_ids = list(st.session_state.grid_ranked["grid_id"].values)
            
            if not st.session_state.get("selected_audit_region"):
                st.session_state["selected_audit_region"] = all_grid_ids[0]
                
            curr_selected = st.session_state["selected_audit_region"]
            default_index = all_grid_ids.index(curr_selected) if curr_selected in all_grid_ids else 0
                
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
            with col_ctrl1:
                target_cell_id = st.selectbox(
                    "Select Area to Audit", 
                    all_grid_ids, 
                    index=default_index, 
                    key="area_audit_select"
                )
            with col_ctrl2:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                audit_btn = st.button("🔎 Audit Selected Area", use_container_width=True, type="primary")
            with col_ctrl3:
                st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
                gen_plan_btn = st.button("📝 Generate Master Recovery Plan", use_container_width=True)
                
            if "last_area_select" not in st.session_state:
                st.session_state["last_area_select"] = target_cell_id
            if target_cell_id != st.session_state["last_area_select"]:
                st.session_state["last_area_select"] = target_cell_id
                st.session_state["selected_audit_region"] = target_cell_id
                st.session_state["show_region_audit"] = True
                st.session_state["show_master_plan"] = False

            if audit_btn:
                st.session_state["show_region_audit"] = True
                st.session_state["show_master_plan"] = False
                st.session_state["selected_audit_region"] = target_cell_id
            elif gen_plan_btn:
                st.session_state["show_master_plan"] = True
                st.session_state["show_region_audit"] = False
                
            if not st.session_state.get("show_master_plan", False) and not st.session_state.get("show_region_audit", False):
                st.session_state["show_region_audit"] = True

            # Direct Output Display Below Controls
            if st.session_state.get("show_region_audit", False):
                curr_region = st.session_state.get("selected_audit_region") or target_cell_id
                matched_rows = st.session_state.grid_ranked[st.session_state.grid_ranked["grid_id"] == curr_region]
                if len(matched_rows) > 0:
                    cell_row = matched_rows.iloc[0]
                    tier_label = str(cell_row.get("priority_label", "Recover Later"))
                    tier_color = str(cell_row.get("tier_color", "#22c55e"))
                    tier_icon = str(cell_row.get("tier_icon", "🟢"))
                    rank_str = str(cell_row.get("priority_rank", "Sector"))
                    
                    b_total = int(cell_row["total_buildings"])
                    b_des = int(cell_row["count_destroyed"])
                    b_maj = int(cell_row["count_major_damage"])
                    b_min = int(cell_row["count_minor_damage"])
                    b_nodmg = int(cell_row["count_no_damage"])
                    pct_dmg = ((b_des + b_maj + b_min) / max(1, b_total)) * 100
                    
                    action = cell_row.get("action", "Routine monitoring")
                    needs = cell_row.get("needs", "Standard monitoring equipment")
                    how_to = cell_row.get("how_to_recover", "Maintain baseline observation")
                    
                    if (b_des + b_maj + b_min) > 0:
                        reason = f"{curr_region} has detected damage impact ({b_des} destroyed, {b_maj} major damage, {b_min} minor damage out of {b_total} structures) and is categorized as {tier_label}."
                    else:
                        if b_total == 0:
                            reason = f"{curr_region} contains 0 assessed buildings and presents minimal structural risk; assigned baseline status."
                        else:
                            reason = f"{curr_region} has all {b_nodmg} buildings undamaged and is assigned routine baseline monitoring."

                    audit_html = f"""
<div style="background: #ffffff; border: 1px solid #cbd5e1; border-left: 6px solid {tier_color}; border-radius: 10px; padding: 1.5rem 1.8rem; box-shadow: 0 4px 12px rgba(0,0,0,0.06); margin-top: 1rem; margin-bottom: 1.5rem;">
    <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #64748b; margin-bottom: 0.3rem;">
        REGIONAL DAMAGE AUDIT — {rank_str.upper()}
    </div>
    <h3 style="margin: 0 0 1.2rem 0; color: #0f172a; font-size: 1.45rem; font-weight: 800;">
        {curr_region} — Detailed Area Audit
    </h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1.2rem; background: #f8fafc; padding: 1rem 1.2rem; border-radius: 8px; border: 1px solid #e2e8f0;">
        <div>
            <div style="font-size: 0.82rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Priority</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: {tier_color}; margin-top: 0.2rem;">{tier_icon} {tier_label}</div>
        </div>
        <div>
            <div style="font-size: 0.82rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Buildings Assessed</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin-top: 0.2rem;">{b_total}</div>
        </div>
        <div>
            <div style="font-size: 0.82rem; color: #64748b; font-weight: 600; text-transform: uppercase;">Damage Requiring Recovery</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #ef4444; margin-top: 0.2rem;">{pct_dmg:.1f}%</div>
        </div>
    </div>
    <div style="font-size: 0.95rem; color: #334155; margin-bottom: 1rem; line-height: 1.6;">
        <b>Structural Damage Breakdown:</b><br>
        <span style="color: #ef4444; font-weight: 600;">🔴 Destroyed: {b_des}</span><br>
        <span style="color: #f97316; font-weight: 600;">🟠 Major Damage: {b_maj}</span><br>
        <span style="color: #ca8a04; font-weight: 600;">🟡 Minor Damage: {b_min}</span><br>
        <span style="color: #16a34a; font-weight: 600;">🟢 Undamaged: {b_nodmg}</span>
    </div>
    <div style="border-top: 1px solid #e2e8f0; padding-top: 0.9rem; margin-bottom: 0.8rem;">
        <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-bottom: 0.25rem;">Required Equipment & Resources:</div>
        <div style="font-size: 0.95rem; color: #334155; line-height: 1.4;">{needs}</div>
    </div>
    <div style="border-top: 1px solid #e2e8f0; padding-top: 0.9rem; margin-bottom: 0.8rem;">
        <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-bottom: 0.25rem;">Recommended Recovery Procedure:</div>
        <div style="font-size: 0.95rem; color: #334155; line-height: 1.4;">{how_to}</div>
    </div>
    <div style="border-top: 1px solid #e2e8f0; padding-top: 0.9rem;">
        <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a; margin-bottom: 0.25rem;">Prioritization Reason:</div>
        <div style="font-size: 0.95rem; color: #334155; line-height: 1.4;">{reason}</div>
    </div>
</div>
"""
                    render_html_block(audit_html)
                else:
                    st.warning(f"Region data for {curr_region} is not available.")
                    
            elif st.session_state.get("show_master_plan", False):
                # Identify which region must be recovered first
                has_dmg = (st.session_state.grid_ranked["priority_score"] > 0)
                damaged_df = st.session_state.grid_ranked[has_dmg]
                
                if len(damaged_df) > 0:
                    top_dmg_row = damaged_df.iloc[0]
                    top_dmg_name = top_dmg_row["grid_id"]
                    top_dmg_des = int(top_dmg_row["count_destroyed"])
                    top_dmg_maj = int(top_dmg_row["count_major_damage"])
                    top_dmg_min = int(top_dmg_row["count_minor_damage"])
                    top_dmg_tot = int(top_dmg_row["total_buildings"])
                    
                    first_recover_html = f"""
<div style="background: #fef2f2; border: 2px solid #ef4444; border-radius: 10px; padding: 1.4rem 1.6rem; margin-top: 1rem; margin-bottom: 1.5rem; box-shadow: 0 4px 14px rgba(239, 68, 68, 0.15);">
    <div style="font-size: 0.85rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; color: #991b1b; margin-bottom: 0.35rem;">
        🚨 PRIMARY RECOVERY FOCUS — WHICH REGION MUST BE RECOVERED FIRST
    </div>
    <h3 style="color: #991b1b; margin: 0 0 0.6rem 0; font-size: 1.5rem; font-weight: 800;">
        🔴 {top_dmg_name} Must Be Recovered FIRST
    </h3>
    <p style="margin: 0 0 0.8rem 0; font-size: 0.98rem; color: #7f1d1d; line-height: 1.5;">
        The region where the damage occurs more is prioritized to be recovered first. <b>{top_dmg_name}</b> sustained the highest cumulative structural damage across the disaster scene with <b>{top_dmg_des} destroyed</b>, <b>{top_dmg_maj} majorly damaged</b>, and <b>{top_dmg_min} minorly damaged</b> structures out of {top_dmg_tot} total assessed buildings.
    </p>
    <div style="background: white; border-radius: 8px; padding: 0.9rem 1.2rem; border-left: 4px solid #ef4444;">
        <b>How {top_dmg_name} Can Be Recovered First:</b><br/>
        <span style="color: #334155; font-size: 0.92rem; line-height: 1.5;">{top_dmg_row.get('how_to_recover', '')}</span><br/><br/>
        <b>What {top_dmg_name} Needs:</b><br/>
        <span style="color: #334155; font-size: 0.92rem; line-height: 1.5;">{top_dmg_row.get('needs', '')}</span>
    </div>
</div>
"""
                    overall_text = f"Recovery begins in {top_dmg_name} (Recover First) due to highest structural damage ({top_dmg_des} destroyed, {top_dmg_maj} major damage). Subsequent sectors are sequenced strictly according to damage concentration."
                else:
                    first_recover_html = f"""
<div style="background: #f0fdf4; border: 2px solid #22c55e; border-radius: 10px; padding: 1.4rem 1.6rem; margin-top: 1rem; margin-bottom: 1.5rem;">
    <h3 style="color: #166534; margin: 0 0 0.5rem 0; font-size: 1.35rem; font-weight: 800;">
        🟢 All Regions Are Undamaged — Baseline Monitoring Mode
    </h3>
    <p style="margin: 0; font-size: 0.98rem; color: #14532d; line-height: 1.5;">
        No critical structural damage was detected across surveyed sectors. All sectors are scheduled for routine municipal baseline inspections.
    </p>
</div>
"""
                    overall_text = "All sectors are undamaged. Baseline environmental monitoring and civil inspection is active."

                render_html_block(first_recover_html)

                # Overall Comprehensive Regions Breakdown
                st.markdown("#### 🗺️ Comprehensive Overall Regions Breakdown")
                regions_html_blocks = []
                for idx, row in st.session_state.grid_ranked.iterrows():
                    r_name = row["grid_id"]
                    tier_label = str(row.get("priority_label", "Recover Later"))
                    tier_color = str(row.get("tier_color", "#22c55e"))
                    tier_icon = str(row.get("tier_icon", "🟢"))
                    rank_str = str(row.get("priority_rank", f"Sector #{idx + 1}"))
                    
                    b_total = int(row["total_buildings"])
                    b_des = int(row["count_destroyed"])
                    b_maj = int(row["count_major_damage"])
                    b_min = int(row["count_minor_damage"])
                    b_nodmg = int(row["count_no_damage"])
                    
                    needs = row.get("needs", "Routine municipal inspection equipment")
                    how_to = row.get("how_to_recover", "Maintain baseline observation")
                    
                    block = f"""
<div style="background: #ffffff; border: 1px solid #cbd5e1; border-left: 6px solid {tier_color}; border-radius: 8px; padding: 1.1rem 1.3rem; margin-bottom: 0.85rem; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
    <div style="font-size: 1.1rem; font-weight: 700; color: #0f172a; margin-bottom: 0.35rem;">
        {tier_icon} <b>{r_name}</b> — <span style="color: {tier_color};">{tier_label} ({rank_str})</span>
    </div>
    <div style="font-size: 0.92rem; color: #334155; margin-bottom: 0.45rem;">
        <b>Buildings Assessed:</b> {b_total} &nbsp;|&nbsp; 
        <span style="color: #ef4444; font-weight: 600;">{b_des} destroyed</span> &nbsp;|&nbsp; 
        <span style="color: #f97316; font-weight: 600;">{b_maj} major</span> &nbsp;|&nbsp; 
        <span style="color: #ca8a04; font-weight: 600;">{b_min} minor</span> &nbsp;|&nbsp; 
        <span style="color: #16a34a; font-weight: 600;">{b_nodmg} undamaged</span>
    </div>
    <div style="font-size: 0.9rem; color: #1e293b; background: #f8fafc; padding: 0.75rem 1rem; border-radius: 6px; margin-bottom: 0.45rem;">
        <b>What It Needs:</b> {needs}
    </div>
    <div style="font-size: 0.9rem; color: #1e293b; background: #f8fafc; padding: 0.75rem 1rem; border-radius: 6px;">
        <b>How It Can Be Recovered:</b> {how_to}
    </div>
</div>
"""
                    regions_html_blocks.append(block)

                render_html_block("".join(regions_html_blocks))

                # PDF Download ONLY (ReportLab - Cached for Instant Download)
                if st.session_state.get("cached_master_plan_pdf") is None:
                    st.session_state["cached_master_plan_pdf"] = generate_recovery_plan_pdf(
                        st.session_state.grid_ranked, 
                        overall_text,
                        pre_image=st.session_state.pre_image,
                        post_image=st.session_state.aligned_post_image
                    )
                st.download_button(
                    label="📄 Download Master Recovery Plan (PDF)",
                    data=st.session_state["cached_master_plan_pdf"],
                    file_name="master_recovery_plan.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.info("👈 Select an area and click **Audit Selected Area** or click **Generate Master Recovery Plan** to view recovery directives.")
        else:
            st.warning("⚠️ Spatial assessment results are not available yet. Please run the Damage Assessment first.")
