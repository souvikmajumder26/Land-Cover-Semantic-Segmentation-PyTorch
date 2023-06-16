from pathlib import Path
from enum import Enum

class Constants(Enum):
    VERSION = "1.0"
    USE_CASE = "Land_Cover_Semantic_Segmentation"
    CONFIG_PATH = Path("config/config.yaml")
    CLASSES = ['background', 'building', 'woodland', 'water', 'road']