from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
IMAGES_DIR = BASE_DIR / "images"

VEHICLE_PEDESTRIAN_MODEL = MODELS_DIR / "vehicle_pedestrian_detection.pt"
SIGN_DETECTION_MODEL = MODELS_DIR / "sign_detection.pt"
SIGN_CLASSIFICATION_MODEL = MODELS_DIR / "traffic_sign_classification.pt"
LIGHT_DETECTION_CLASSIFICATION_MODEL = MODELS_DIR / "traffic_light_detect_class.pt"
