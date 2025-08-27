from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
IMAGES_DIR = BASE_DIR / "images"
VIDEOS_DIR = BASE_DIR / "videos"

VEHICLE_PEDESTRIAN_MODEL = MODELS_DIR / "vehicle_pedestrian_detection.pt"
SIGN_DETECTION_MODEL = MODELS_DIR / "sign_detection.pt"
SIGN_CLASSIFICATION_MODEL = MODELS_DIR / "traffic_sign_classification.h5"
LIGHT_DETECTION_CLASSIFICATION_MODEL = MODELS_DIR / "traffic_light_detect_class.pt"


BEAMNG_HOME = r'C:\Users\user\Documents\beamng-tech\BeamNG.tech.v0.36.4.0'
