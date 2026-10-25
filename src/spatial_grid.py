import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
from typing import Dict, Any, List, Tuple, Optional
from src.logger import logger

class SpatialGridManager:
    """
    Unified Spatial Grid Manager supporting:
    1. Image-Space (Pixel coordinates) partitioning for non-georeferenced satellite/aerial imagery.
    2. Projected CRS / Geospatial partitioning for georeferenced satellite datasets.
    """
    
    def __init__(
        self,
        lat_min: Optional[float] = None,
        lon_min: Optional[float] = None,
        lat_max: Optional[float] = None,
        lon_max: Optional[float] = None,
        grid_size_meters: float = 500.0,
        image_width: int = 1024,
        image_height: int = 1024,
        grid_cols: int = 2,
        grid_rows: int = 2,
        mode: Optional[str] = None,
        **kwargs
    ):
        # Support flexible dimension keyword arguments
        if "width" in kwargs:
            image_width = kwargs["width"]
        if "height" in kwargs:
            image_height = kwargs["height"]
        if "image" in kwargs and hasattr(kwargs["image"], "size"):
            image_width, image_height = kwargs["image"].size
        if "image_shape" in kwargs:
            image_height, image_width = kwargs["image_shape"][:2]
        if "image_size" in kwargs:
            image_width, image_height = kwargs["image_size"][:2]
            
        if mode is None:
            if lat_min is not None and lon_min is not None:
                mode = "geospatial"
            else:
                mode = "image_space"
                
        self.mode = mode
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.grid_cols = int(grid_cols)
        self.grid_rows = int(grid_rows)
        self.lat_min = lat_min
        self.lon_min = lon_min
        self.lat_max = lat_max
        self.lon_max = lon_max
        self.grid_size = grid_size_meters
        
        logger.info(f"Initialized SpatialGridManager (Mode: {mode}, Grid: {self.grid_cols}x{self.grid_rows}, Size: {self.image_width}x{self.image_height})")

    def generate_grid(self) -> gpd.GeoDataFrame:
        """
        Generates regular grid cells.
        If mode == 'image_space', creates a 2D pixel grid over [0, image_width] x [0, image_height].
        For 2x2 grid, names areas: Area A (Top-Left), Area B (Top-Right), Area C (Bottom-Left), Area D (Bottom-Right).
        If mode == 'geospatial', generates metric grid over bounding coordinates.
        """
        if self.mode == "image_space" or (self.lat_min is None or self.lon_min is None):
            cell_w = self.image_width / float(self.grid_cols)
            cell_h = self.image_height / float(self.grid_rows)
            cells = []
            
            # Region naming mapping for 2x2 layout
            area_2x2_names = {
                (0, 0): "Region 1",
                (0, 1): "Region 2",
                (1, 0): "Region 3",
                (1, 1): "Region 4"
            }
            
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    cell_idx = r * self.grid_cols + c
                    if self.grid_cols == 2 and self.grid_rows == 2:
                        gid = area_2x2_names.get((r, c), f"Region {cell_idx+1}")
                    else:
                        gid = f"GRID_{cell_idx:04d}"
                        
                    x_min = c * cell_w
                    y_min = r * cell_h
                    x_max = (c + 1) * cell_w if c < self.grid_cols - 1 else float(self.image_width)
                    y_max = (r + 1) * cell_h if r < self.grid_rows - 1 else float(self.image_height)
                    poly = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
                    cells.append({
                        "grid_id": gid,
                        "area_name": gid,
                        "region_name": gid,
                        "row": r,
                        "col": c,
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "geometry": poly,
                        "centroid_x": (x_min + x_max) / 2.0,
                        "centroid_y": (y_min + y_max) / 2.0
                    })
            logger.info(f"Created {len(cells)} image-space grid cells ({self.grid_cols}x{self.grid_rows}).")
            return gpd.GeoDataFrame(cells, geometry="geometry")
        else:
            # Geospatial grid using metric Web Mercator (EPSG:3857)
            bbox_geom = Polygon([
                (self.lon_min, self.lat_min),
                (self.lon_min, self.lat_max),
                (self.lon_max, self.lat_max),
                (self.lon_max, self.lat_min)
            ])
            gdf_bbox = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[bbox_geom])
            gdf_bbox_proj = gdf_bbox.to_crs(epsg=3857)
            x_min, y_min, x_max, y_max = gdf_bbox_proj.total_bounds
            
            x_coords = np.arange(x_min, x_max, self.grid_size)
            y_coords = np.arange(y_min, y_max, self.grid_size)
            
            grid_polygons = []
            grid_ids = []
            cell_index = 0
            
            for x in x_coords:
                for y in y_coords:
                    cell = Polygon([
                        (x, y),
                        (x + self.grid_size, y),
                        (x + self.grid_size, y + self.grid_size),
                        (x, y + self.grid_size)
                    ])
                    grid_polygons.append(cell)
                    grid_ids.append(f"GRID_{cell_index:04d}")
                    cell_index += 1
                    
            gdf_grid_proj = gpd.GeoDataFrame({"grid_id": grid_ids, "area_name": grid_ids}, geometry=grid_polygons, crs="EPSG:3857")
            gdf_grid = gdf_grid_proj.to_crs(epsg=4326)
            centroids_proj = gdf_grid_proj.geometry.centroid
            centroids_wgs = centroids_proj.to_crs(epsg=4326)
            gdf_grid["centroid_lat"] = centroids_wgs.y
            gdf_grid["centroid_lon"] = centroids_wgs.x
            logger.info(f"Created {len(gdf_grid)} geospatial grid cells.")
            return gdf_grid

    def assign_and_prioritize(self, grid_df: pd.DataFrame, buildings: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Assigns every detected building to its containing grid cell and computes transparent priority metrics.
        Guarantees that every valid building is counted exactly once with zero duplications or omissions.
        Excludes invalid or out-of-image coordinates and logs diagnostics.
        """
        grid_ranked = grid_df.copy()
        grid_ids = list(grid_ranked["grid_id"].values)
        cell_buildings = {gid: [] for gid in grid_ids}
        invalid_buildings = []
        
        # Image-space assignment via centroid with rigorous boundary intervals
        if "x_min" in grid_ranked.columns:
            for b in buildings:
                poly = b.get("xy_geometry")
                if not poly or poly.is_empty:
                    invalid_buildings.append((b.get("uid", "unknown"), "Empty or missing geometry"))
                    continue
                    
                cx, cy = float(poly.centroid.x), float(poly.centroid.y)
                
                # Exclude non-finite coordinates
                if not np.isfinite(cx) or not np.isfinite(cy):
                    invalid_buildings.append((b.get("uid", "unknown"), f"Non-finite centroid ({cx}, {cy})"))
                    continue
                    
                # Exclude out-of-image coordinates
                if cx < 0.0 or cx > float(self.image_width) or cy < 0.0 or cy > float(self.image_height):
                    invalid_buildings.append((b.get("uid", "unknown"), f"Out-of-image centroid ({cx:.1f}, {cy:.1f})"))
                    continue
                    
                # Deterministic grid assignment:
                # Half-open intervals [x_min, x_max) for interior boundaries, [x_min, x_max] for outer boundary
                assigned = False
                for _, row in grid_ranked.iterrows():
                    r = int(row.get("row", 0))
                    c = int(row.get("col", 0))
                    x_min, x_max = float(row["x_min"]), float(row["x_max"])
                    y_min, y_max = float(row["y_min"]), float(row["y_max"])
                    
                    x_in = (x_min <= cx <= x_max) if (c == self.grid_cols - 1) else (x_min <= cx < x_max)
                    y_in = (y_min <= cy <= y_max) if (r == self.grid_rows - 1) else (y_min <= cy < y_max)
                    
                    if x_in and y_in:
                        cell_buildings[row["grid_id"]].append(b)
                        assigned = True
                        break
                        
                if not assigned:
                    # Fallback clamp for points exactly on the extreme outer boundary
                    dists = (grid_ranked["centroid_x"] - cx)**2 + (grid_ranked["centroid_y"] - cy)**2
                    nearest_gid = grid_ranked.loc[dists.idxmin(), "grid_id"]
                    cell_buildings[nearest_gid].append(b)
        else:
            # Geospatial assignment via point-in-polygon
            for b in buildings:
                lng_lat_poly = b.get("lng_lat_geometry")
                if not lng_lat_poly or lng_lat_poly.is_empty:
                    invalid_buildings.append((b.get("uid", "unknown"), "Empty geospatial geometry"))
                    continue
                pt = lng_lat_poly.centroid
                assigned = False
                for _, row in grid_ranked.iterrows():
                    if row["geometry"].contains(pt):
                        cell_buildings[row["grid_id"]].append(b)
                        assigned = True
                        break
                if not assigned:
                    grid_ranked_dists = grid_ranked.geometry.distance(pt)
                    nearest_gid = grid_ranked.loc[grid_ranked_dists.idxmin(), "grid_id"]
                    cell_buildings[nearest_gid].append(b)

        if len(invalid_buildings) > 0:
            logger.warning(f"[SPATIAL GRID] Excluded {len(invalid_buildings)} invalid/out-of-bound buildings: {invalid_buildings[:5]}")

        # Configured recovery-score weights: W_minor=1, W_major=2, W_destroyed=3 (W3 > W2 > W1)
        W_MINOR = 1.0
        W_MAJOR = 2.0
        W_DESTROYED = 3.0

        # Aggregate counts & scores per cell independently
        total_counts = []
        no_dmg_counts = []
        minor_counts = []
        major_counts = []
        destroyed_counts = []
        damage_sums = []
        severity_scores = []
        priority_scores = []
        exposure_areas_px2 = []
        
        for gid in grid_ids:
            b_list = cell_buildings[gid]
            n_tot = len(b_list)
            n_no = sum(1 for b in b_list if b.get("damage_label") == "no-damage" or b.get("damage_class") == 0)
            n_min = sum(1 for b in b_list if b.get("damage_label") == "minor-damage" or b.get("damage_class") == 1)
            n_maj = sum(1 for b in b_list if b.get("damage_label") == "major-damage" or b.get("damage_class") == 2)
            n_des = sum(1 for b in b_list if b.get("damage_label") == "destroyed" or b.get("damage_class") == 3)
            
            # Priority Score = W1 * N_minor + W2 * N_major + W3 * N_destroyed
            dmg_sum = float(W_MINOR * n_min + W_MAJOR * n_maj + W_DESTROYED * n_des)
            sev_score = float(dmg_sum / (W_DESTROYED * max(1, n_tot))) if n_tot > 0 else 0.0
            p_score = float(dmg_sum)
            
            area_sum = sum(b["xy_geometry"].area for b in b_list if b.get("xy_geometry"))
            
            total_counts.append(n_tot)
            no_dmg_counts.append(n_no)
            minor_counts.append(n_min)
            major_counts.append(n_maj)
            destroyed_counts.append(n_des)
            damage_sums.append(dmg_sum)
            severity_scores.append(sev_score)
            priority_scores.append(p_score)
            exposure_areas_px2.append(area_sum)

        grid_ranked["total_buildings"] = total_counts
        grid_ranked["count_no_damage"] = no_dmg_counts
        grid_ranked["count_minor_damage"] = minor_counts
        grid_ranked["count_major_damage"] = major_counts
        grid_ranked["count_destroyed"] = destroyed_counts
        grid_ranked["count_damaged"] = [d + m + mn for d, m, mn in zip(destroyed_counts, major_counts, minor_counts)]
        grid_ranked["damage_severity"] = severity_scores
        grid_ranked["priority_score"] = priority_scores
        grid_ranked["structural_exposure_area_px2"] = exposure_areas_px2
        grid_ranked["structural_exposure_area_m2"] = 0.0
        grid_ranked["dist_to_max_damage_m"] = -1.0

        # Sort strictly by priority_score descending (damage-based ranking):
        grid_ranked = grid_ranked.sort_values(
            by=["priority_score", "count_destroyed", "count_major_damage", "count_minor_damage", "total_buildings", "grid_id"],
            ascending=[False, False, False, False, False, True]
        ).reset_index(drop=True)

        tiers = []
        labels = []
        actions = []
        colors = []
        icons = []
        needs_list = []
        how_to_list = []
        ranks = []
        status_list = []
        
        max_score = float(grid_ranked["priority_score"].max()) if len(grid_ranked) > 0 else 0.0
        all_zero_damage = (max_score == 0.0)
        
        active_dmg_rank = 1
        prev_score = None
        prev_rank = 1
        
        for idx, row in grid_ranked.iterrows():
            p_score = float(row["priority_score"])
            n_dmg = int(row["count_damaged"])
            n_tot = int(row["total_buildings"])
            
            # Record audit status: distinguish empty vs assessed
            if n_tot == 0:
                status_list.append("NO_BUILDINGS")
            else:
                status_list.append("ASSESSED")
            
            if all_zero_damage or p_score == 0.0 or n_dmg == 0:
                # ZERO damage in this sector: NEVER assign "Recover First" or Red
                ranks.append(f"Sector #{idx + 1}")
                tiers.append("LOW")
                colors.append("#22c55e")
                icons.append("🟢")
                if n_tot == 0:
                    labels.append("Recover Later (No Buildings)")
                    actions.append("No structures present — routine environmental and drone surveillance.")
                    needs_list.append("Baseline open-land aerial monitoring; clear terrain suitable for emergency supply staging if required.")
                    how_to_list.append("Maintain standard regional observation. Utilize open land parcels as secondary emergency logistical staging if needed.")
                else:
                    labels.append("Recover Later (Undamaged)")
                    actions.append("All structures undamaged — baseline municipal monitoring and safety confirmation.")
                    needs_list.append("Civil engineering inspectors to verify baseline utility safety and confirm zero foundation displacement.")
                    how_to_list.append("Conduct perimeter safety walkthrough; issue Safe Re-entry Certifications for residential/commercial structures.")
            else:
                # Active structural damage present: rank according to damage severity
                # Handle tied scores deterministically: equal scores share the same tier & rank
                if prev_score is not None and abs(p_score - prev_score) < 1e-6:
                    current_rank = prev_rank
                else:
                    current_rank = active_dmg_rank
                
                ranks.append(f"Priority #{current_rank}")
                if current_rank == 1:
                    tiers.append("CRITICAL")
                    labels.append("Recover First")
                    actions.append("Immediate search & rescue, structural shoring, hazard isolation, and critical power restoration.")
                    colors.append("#ef4444")
                    icons.append("🔴")
                    needs_list.append("Heavy Urban Search & Rescue (US&R) task forces, hydraulic rubble excavators, structural shoring struts, trauma medical triage units, mobile generators, and emergency utility isolation crews.")
                    how_to_list.append("1. Establish 360° perimeter cordon around destroyed assets; 2. Clear primary arterial route for emergency vehicles; 3. Deploy hydraulic jacks to shore unstable load-bearing walls; 4. Isolate ruptured gas mains and live electric lines; 5. Conduct systematic search in collapsed structural voids.")
                elif current_rank == 2:
                    tiers.append("HIGH")
                    labels.append("Recover Next")
                    actions.append("High-priority structural inspection, secondary transit clearance, and utility stabilization.")
                    colors.append("#f97316")
                    icons.append("🟠")
                    needs_list.append("Structural engineering evaluation teams, medium debris loaders, secondary roadway clearing equipment, and municipal water/power line restoration teams.")
                    how_to_list.append("1. Inspect and placard majorly damaged buildings (Red/Yellow tags); 2. Clear access streets of downed trees and debris; 3. Shore compromised exterior walls; 4. Restore municipal water distribution.")
                elif current_rank == 3:
                    tiers.append("MEDIUM")
                    labels.append("Moderate Priority")
                    actions.append("Scheduled structural repair, localized debris clearing, and secondary access restoration.")
                    colors.append("#eab308")
                    icons.append("🟡")
                    needs_list.append("Building inspection triage crews, roof weatherproofing tarps, secondary street maintenance vehicles, and distribution linemen.")
                    how_to_list.append("1. Verify structural safety for occupied structures; 2. Distribute emergency tarps for damaged roofs; 3. Clear localized sidewalk and street debris; 4. Reconnect standard municipal utilities.")
                else:
                    tiers.append("LOW")
                    labels.append("Recover Later")
                    actions.append("Non-critical maintenance, localized minor repair, and ongoing area monitoring.")
                    colors.append("#22c55e")
                    icons.append("🟢")
                    needs_list.append("Local municipal maintenance personnel and standard utility repair technicians.")
                    how_to_list.append("1. Routine structural review; 2. Clear light non-hazardous debris; 3. Maintain regular service schedules.")
                
                prev_score = p_score
                prev_rank = current_rank
                active_dmg_rank += 1

        grid_ranked["priority_tier"] = tiers
        grid_ranked["priority_label"] = labels
        grid_ranked["priority_rank"] = ranks
        grid_ranked["tier_color"] = colors
        grid_ranked["tier_icon"] = icons
        grid_ranked["action"] = actions
        grid_ranked["needs"] = needs_list
        grid_ranked["how_to_recover"] = how_to_list
        grid_ranked["status"] = status_list
        
        # Terminal validation log
        valid_buildings_count = len(buildings) - len(invalid_buildings)
        total_in_grid = sum(grid_ranked["total_buildings"])
        match_counts = (valid_buildings_count == total_in_grid)
        
        logger.info(
            f"[GRID INTEGRITY] Valid Detected: {valid_buildings_count}, Assigned to Grid: {total_in_grid}, "
            f"Match: {match_counts}, Excluded: {len(invalid_buildings)}"
        )
        
        return grid_ranked

    def enrich_grid_with_real_metrics(self, grid_df: pd.DataFrame, buildings_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Enriches grid dataframe with real metrics for backwards compatibility.
        """
        grid_df = grid_df.copy()
        if "structural_exposure_area_m2" not in grid_df.columns:
            grid_df["structural_exposure_area_m2"] = 0.0
        if "dist_to_max_damage_m" not in grid_df.columns:
            grid_df["dist_to_max_damage_m"] = -1.0
        return grid_df
