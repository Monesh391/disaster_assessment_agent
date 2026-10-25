import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
from shapely.geometry import Polygon
from src.logger import logger

# Modern HSL-derived color palette for damage visualization
DAMAGE_COLORS = {
    "no-damage": "#10B981",       # Emerald Green
    "minor-damage": "#F59E0B",    # Amber Orange
    "major-damage": "#EF4444",    # Rose Red
    "destroyed": "#8B5CF6",       # Violet/Purple
    "un-classified": "#64748B"    # Slate Gray
}

def plot_xbd_pair(
    pre_img_path: str,
    post_img_path: str,
    buildings: List[Dict[str, Any]],
    pair_id: str = "Image Pair",
    disaster_name: str = "Disaster"
) -> plt.Figure:
    """
    Generates a side-by-side Matplotlib visualization of pre- and post-disaster satellite images.
    Overlays building polygons with boundary lines and transparent fills color-coded by damage.
    """
    logger.info(f"Generating Matplotlib visualization for image pair: {pair_id}")
    
    pre_img = Image.open(pre_img_path)
    post_img = Image.open(post_img_path)
    
    # Create figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=100)
    fig.suptitle(f"xBD Damage Assessment - {pair_id} ({disaster_name.replace('-', ' ').title()})", 
                 fontsize=14, fontweight="bold", y=0.98)
    
    # Left subplot: Pre-Disaster
    ax1.imshow(pre_img)
    ax1.set_title("Pre-Disaster Imagery (Reference)", fontsize=11, fontweight="semibold")
    ax1.axis("off")
    
    # Right subplot: Post-Disaster
    ax2.imshow(post_img)
    ax2.set_title("Post-Disaster Imagery (Damage Assessed)", fontsize=11, fontweight="semibold")
    ax2.axis("off")
    
    # Draw building footprints
    for b in buildings:
        poly = b["xy_geometry"]
        damage = b["damage_label"]
        color = DAMAGE_COLORS.get(damage, DAMAGE_COLORS["un-classified"])
        
        if isinstance(poly, Polygon) and not poly.is_empty:
            # Extract exterior ring coordinates
            x, y = poly.exterior.xy
            coords = list(zip(x, y))
            
            # Draw building footprint boundary on pre-disaster (neutral white line)
            patch_pre = patches.Polygon(
                coords, 
                closed=True, 
                edgecolor="white", 
                facecolor="none", 
                linewidth=0.7, 
                alpha=0.6
            )
            ax1.add_patch(patch_pre)
            
            # Draw building footprint with damage color on post-disaster
            patch_post = patches.Polygon(
                coords, 
                closed=True, 
                edgecolor=color, 
                facecolor=color, 
                linewidth=1.2, 
                alpha=0.35
            )
            ax2.add_patch(patch_post)
            
    # Add Legend to the post-disaster plot
    legend_patches = [
        patches.Patch(color=DAMAGE_COLORS["no-damage"], label="No Damage"),
        patches.Patch(color=DAMAGE_COLORS["minor-damage"], label="Minor Damage"),
        patches.Patch(color=DAMAGE_COLORS["major-damage"], label="Major Damage"),
        patches.Patch(color=DAMAGE_COLORS["destroyed"], label="Destroyed"),
        patches.Patch(color=DAMAGE_COLORS["un-classified"], label="Unclassified")
    ]
    ax2.legend(handles=legend_patches, loc="upper right", framealpha=0.8, fontsize=9)
    
    plt.tight_layout()
    return fig
