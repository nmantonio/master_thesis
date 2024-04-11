import os

TFM_PATH = r"/home/anadal/Experiments/TFM"
GIT_PATH = os.path.join(TFM_PATH, "master_thesis")
TRAIN_PATH = os.path.join(TFM_PATH, "trains")

RAW_DATABASE_PATH = r""
DATABASE_PATH = os.path.join(TFM_PATH, "PROCESSED_DATABASE")
CROPPED_DATABASE_PATH = os.path.join(TFM_PATH, "CROPPED_DATABASE")
RAW_MASKS_PATH = os.path.join(TFM_PATH, "MASKS")
MASKS_PATH = os.path.join(TFM_PATH, "PROCESSED_MASKS")

DETECTION_CSV = os.path.join(GIT_PATH, "abnormal_detection_selection.csv")
CLASSIFICATION_CSV = os.path.join(GIT_PATH, "disease_classification_selection.csv")
BBOX_CSV = os.path.join(GIT_PATH, "XAI", "pulmonary_fibrosis_BB.csv")

ENCODERS_PATH = os.path.join(GIT_PATH, "encoders")
DETECTION_ENCODER = os.path.join(ENCODERS_PATH, "detection_encoder")
CLASSIFICATION_ENCODER = os.path.join(ENCODERS_PATH, "classification_encoder")

AUGMENTED_IMAGES_CHECK = os.path.join(TFM_PATH, "AUGMENTED_IMAGES")
IMAGES_CHECK = os.path.join(TFM_PATH, "TEMP_IMAGES")
os.makedirs(IMAGES_CHECK, exist_ok=True)

TRAIN_SHEET = os.path.join(GIT_PATH, "train_sheet.xlsx")
RESULTS_SHEET = os.path.join(GIT_PATH, "results_sheet.xlsx")