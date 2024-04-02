import os

TFM_PATH = r"/home/anadal/Experiments/TFM"
GIT_PATH = os.path.join(TFM_PATH, "master_thesis")

RAW_DATABASE_PATH = r""
DATABASE_PATH = os.path.join(TFM_PATH, "PROCESSED_DATABASE")
MASKS_PATH = r""

DETECTION_CSV = os.path.join(GIT_PATH, "abnormal_detection_selection.csv")
CLASSIFICATION_CSV = os.path.join(GIT_PATH, "disease_classification_selection.csv")

ENCODERS_PATH = os.path.join(GIT_PATH, "encoders")
DETECTION_ENCODER = os.path.join(ENCODERS_PATH, "detection_encoder")
CLASSIFICATION_ENCODER = os.path.join(ENCODERS_PATH, "classification_encoder")