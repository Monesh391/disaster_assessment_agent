import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict

# Load environment variables from .env file
load_dotenv()

class AppConfig:
    """
    Configuration helper class to load and expose yaml configuration
    and environment variables.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        # Resolve config path relative to project root
        self.project_root = Path(__file__).resolve().parent.parent
        self.config_file = self.project_root / config_path
        
        self.settings: Dict[str, Any] = {}
        self.load_yaml_config()
        
    def load_yaml_config(self) -> None:
        """Loads configuration from YAML file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.settings = yaml.safe_load(f) or {}
            except Exception as e:
                # If logger is not initialized yet, print to stderr
                import sys
                print(f"Error loading configuration file {self.config_file}: {e}", file=sys.stderr)
                self.settings = {}
        else:
            # Fallback values if configuration file is missing
            self.settings = {
                "map": {
                    "center_lat": 25.7617,
                    "center_lon": -80.1918,
                    "default_zoom": 12,
                    "bounding_box": {"lat_min": 25.70, "lon_min": -80.28, "lat_max": 25.82, "lon_max": -80.12}
                },
                "grid": {"default_size_meters": 1000.0, "srid": 3857},
                "ranking": {
                    "weights": {
                        "damage_severity": 0.40,
                        "population_density": 0.25,
                        "hospital_proximity": 0.20,
                        "critical_infrastructure": 0.15
                    }
                },
                "agent": {
                    "default_provider": "gemini",
                    "gemini": {"model_name": "gemini-1.5-flash", "temperature": 0.2},
                    "openai": {"model_name": "gpt-4o-mini", "temperature": 0.2}
                }
            }

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Retrieves a nested configuration value using dot notation (e.g. 'map.center_lat').
        """
        keys = key_path.split('.')
        current = self.settings
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    @property
    def gemini_api_key(self) -> str:
        return os.getenv("GEMINI_API_KEY", "")

    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def log_level(self) -> str:
        return os.getenv("LOG_LEVEL", "INFO")

# Singleton configuration instance
config = AppConfig()
