import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, Any, List
from src.logger import logger
from src.damage_assessment import GeospatialHazardDamageAssessor

class PriorityRankingEngine:
    """
    Implements Multi-Criteria Decision Analysis (MCDA) to rank grid cells
    for disaster recovery prioritization based on dynamic parameters.
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        # Fallback default weights if none are provided
        self.weights = weights or {
            "damage_severity": 0.40,
            "population_density": 0.25,
            "hospital_proximity": 0.20,
            "critical_infrastructure": 0.15
        }
        
        # Verify weights sum up to ~1.0
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0, atol=1e-3):
            logger.warning(f"Weights do not sum to 1.0 (Sum: {total_weight}). Normalizing weights.")
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def simulate_hazard_damage(self, grid_gdf: gpd.GeoDataFrame, center_lat: float, center_lon: float, hazard_type: str = "wind", intensity: float = 120.0) -> gpd.GeoDataFrame:
        """
        Simulates physical damage across grid cells based on distance from hazard epicenter
        using physical decay equations and vulnerability curves.
        
        Args:
            grid_gdf (GeoDataFrame): Spatial grid.
            center_lat, center_lon (float): Epicenter of the disaster.
            hazard_type (str): 'wind' (hurricane) or 'flood'.
            intensity (float): Max intensity (wind speed in mph, or flood depth in feet).
            
        Returns:
            GeoDataFrame: Updated grid with physical hazard readings and damage severity scores.
        """
        logger.info(f"Simulating physical hazard damage. Type: {hazard_type}, Peak Intensity: {intensity}")
        
        damage_scores = []
        hazard_values = []
        
        for _, row in grid_gdf.iterrows():
            # Approximate distance in km using equirectangular projection approximation for speed
            lat_dist = (row["centroid_lat"] - center_lat) * 111.0
            lon_dist = (row["centroid_lon"] - center_lon) * 111.0 * np.cos(np.radians(center_lat))
            distance_km = np.sqrt(lat_dist**2 + lon_dist**2)
            
            if hazard_type == "wind":
                # Wind speed decay: exponential decay over distance
                decay_coeff = 0.04
                cell_wind_speed = intensity * np.exp(-decay_coeff * distance_km)
                # Ensure it doesn't decay below ambient 10mph
                cell_wind_speed = max(cell_wind_speed, 10.0)
                hazard_values.append(float(cell_wind_speed))
                
                # Assess damage using building vulnerability curves (weighted average for residential and critical)
                res_damage = GeospatialHazardDamageAssessor.calculate_wind_damage(cell_wind_speed, "residential")
                crit_damage = GeospatialHazardDamageAssessor.calculate_wind_damage(cell_wind_speed, "critical_infrastructure")
                
                # Grid cell average damage
                avg_damage = 0.8 * res_damage + 0.2 * crit_damage
                damage_scores.append(float(avg_damage))
                
            elif hazard_type == "flood":
                # Flood depth decay based on distance from water body/epicenter (assuming depression center)
                decay_coeff = 0.15
                cell_flood_depth = intensity - (decay_coeff * distance_km * 3.28084) # 3.28 ft per meter
                cell_flood_depth = max(cell_flood_depth, 0.0) # no negative flood depth
                hazard_values.append(float(cell_flood_depth))
                
                res_damage = GeospatialHazardDamageAssessor.calculate_flood_damage(cell_flood_depth, "residential")
                crit_damage = GeospatialHazardDamageAssessor.calculate_flood_damage(cell_flood_depth, "critical_infrastructure")
                
                avg_damage = 0.7 * res_damage + 0.3 * crit_damage
                damage_scores.append(float(avg_damage))
                
            else:
                raise ValueError(f"Unknown hazard type: {hazard_type}")
                
        grid_gdf["hazard_value"] = hazard_values
        grid_gdf["damage_severity"] = damage_scores
        
        return grid_gdf

    def calculate_priority_scores(self, grid_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Runs the Multi-Criteria Decision Analysis (MCDA) on the enriched grid.
        
        Applies min-max scaling to inputs:
          - damage_severity (already normalized 0-1)
          - population_density
          - hospital_proximity (closer is prioritized -> inverse distance)
          - critical_infrastructure_count
        """
        logger.info("Calculating dynamic prioritization rankings.")
        
        # Guard against empty dataset
        if grid_gdf.empty:
            logger.warning("Empty GeoDataFrame passed to priority ranking.")
            return grid_gdf

        df = grid_gdf.copy()
        
        # 1. Normalize Population Density (Min-Max)
        pop_min = df["population_density"].min()
        pop_max = df["population_density"].max()
        pop_range = (pop_max - pop_min) if pop_max != pop_min else 1.0
        df["norm_population"] = (df["population_density"] - pop_min) / pop_range
        
        # 2. Normalize Hospital Proximity (Closer = Higher Priority)
        dist_min = df["dist_to_hospital_km"].min()
        dist_max = df["dist_to_hospital_km"].max()
        dist_range = (dist_max - dist_min) if dist_max != dist_min else 1.0
        # Invert: closest hospital gets 1.0, furthest gets 0.0
        df["norm_hospital_proximity"] = 1.0 - ((df["dist_to_hospital_km"] - dist_min) / dist_range)
        
        # 3. Normalize Critical Infrastructure Count
        infra_min = df["critical_infra_count"].min()
        infra_max = df["critical_infra_count"].max()
        infra_range = (infra_max - infra_min) if infra_max != infra_min else 1.0
        df["norm_infra"] = (df["critical_infra_count"] - infra_min) / infra_range
        
        # Ensure damage_severity is in df and clean
        if "damage_severity" not in df.columns:
            df["damage_severity"] = 0.0
            
        # 4. Multi-Criteria Aggregation
        df["priority_score"] = (
            self.weights["damage_severity"] * df["damage_severity"] +
            self.weights["population_density"] * df["norm_population"] +
            self.weights["hospital_proximity"] * df["norm_hospital_proximity"] +
            self.weights["critical_infrastructure"] * df["norm_infra"]
        )
        
        # 5. Assign Tiers (by Percentiles)
        scores = df["priority_score"].values
        q_critical = np.percentile(scores, 85) # Top 15%
        q_high = np.percentile(scores, 60)     # Next 25%
        q_medium = np.percentile(scores, 25)   # Next 35%
        
        def assign_tier(score: float) -> str:
            if score >= q_critical:
                return "CRITICAL"
            elif score >= q_high:
                return "HIGH"
            elif score >= q_medium:
                return "MEDIUM"
            else:
                return "LOW"
                
        df["priority_tier"] = df["priority_score"].apply(assign_tier)
        
        # Sort values
        df = df.sort_values(by="priority_score", ascending=False)
        
        logger.info("Prioritization calculations complete.")
        return df
