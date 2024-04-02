import os

TFM_PATH = r"/home/anadal/Experiments/TFM"
GIT_PATH = r"/home/anadal/Experiments/TFM/master_thesis"

RAW_DATABASE_PATH = r""
DATABASE_PATH = r"/home/anadal/Experiments/TFM/PROCESSED_DATABASE"
MASKS_PATH = r""

DETECTION_CSV = r"/home/anadal/Experiments/TFM/master_thesis/abnormal_detection_selection.csv"
CLASSIFICATION_CSV = r"/home/anadal/Experiments/TFM/master_thesis/disease_classification_selection.csv"

ENCODERS_PATH = r"/home/anadal/Experiments/TFM/master_thesis/encoders"
DETECTION_ENCODER = os.path.join(ENCODERS_PATH, "detection_encoder")
CLASSIFICATION_ENCODER = os.path.join(ENCODERS_PATH, "classification_encoder")