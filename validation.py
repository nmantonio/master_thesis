import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import argparse

parser = argparse.ArgumentParser(description="Scrip for performing validation given training folder")
parser.add_argument('--train_path', type=str, required=True, help='Path storing the model to evaluate')
args = parser.parse_args()

train_path = r"{}".format(args.train_path)
save_path = os.path.join(train_path, "val")
os.makedirs(save_path, exist_ok=True)

from paths import *
from utils import load_encoder
from models.utils import get_preprocessing_func


from keras.models import load_model
import pandas as pd
from natsort import natsorted
import cv2
import numpy as np

import json


with open(os.path.join(train_path, "train_args.json"), 'r') as json_file:
    data = json.load(json_file)

model_name = data["model_name"]
fold_idx = data["fold_idx"]
task = data["task"]
database = data["database"]
if database == "cropped":
    database = CROPPED_DATABASE_PATH
else: 
    database = DATABASE_PATH

preprocessing = get_preprocessing_func(model_name)

if task == "detection":
    encoder = load_encoder(DETECTION_ENCODER)
    df = pd.read_csv(DETECTION_CSV)
elif task == "classification":
    encoder = load_encoder(CLASSIFICATION_ENCODER)
    df = pd.read_csv(CLASSIFICATION_CSV)
else: 
    raise ValueError(f"Task not supported. Try 'detection' or 'classification'.")


model = load_model(os.path.join(train_path, "model.keras"), compile=False)
df = df[df["split"] == str(fold_idx)]

val_df = pd.DataFrame(columns = ["new_filename", "true", "pred"])

predictions = {}
for idx, row in df.iterrows():
    
    image = cv2.imread(os.path.join(database, row["new_filename"]), 0)
    image = np.expand_dims(image, axis=-1)

    image = preprocessing(image)
    image = np.expand_dims(image, axis=0)

    image_pred = model.predict(image, verbose=0)[0]
    
    one_hot_pred = np.zeros_like(image_pred)
    one_hot_pred[np.argmax(image_pred)] = 1
    predicted_label = encoder.inverse_transform(one_hot_pred.reshape(1, -1))
    
    predictions[row["new_filename"]]  = {
        "raw": image_pred.tolist(), 
        "one_hot": one_hot_pred.tolist(), 
        "pred_label": predicted_label[0][0], 
        "true_label": row[task]
    }
    
    val_df = pd.concat([val_df, pd.DataFrame({"new_filename": [row["new_filename"]], "true": [row[task]], "pred": [predicted_label[0][0]]})])

# Store results
val_df.to_csv(os.path.join(save_path, "validation_results.csv"), index=False)
with open(os.path.join(save_path, "validation_raw.json"), "w") as json_file: 
    json.dump(predictions, json_file, indent=4)
    
# Numeric metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
metrics = {
    "accuracy": accuracy_score(y_true=val_df["true"], y_pred=val_df["pred"]), 
    "recall_macro": recall_score(y_true=val_df["true"], y_pred=val_df["pred"], average="macro"), 
    "precision_macro": precision_score(y_true=val_df["true"], y_pred=val_df["pred"], average="macro"), 
    "f1_score_macro": f1_score(y_true=val_df["true"], y_pred=val_df["pred"], average="macro"),
    "recall_weighted": recall_score(y_true=val_df["true"], y_pred=val_df["pred"], average="weighted"), 
    "precision_weighted": precision_score(y_true=val_df["true"], y_pred=val_df["pred"], average="weighted"), 
    "f1_score_weighted": f1_score(y_true=val_df["true"], y_pred=val_df["pred"], average="weighted")
}
with open(os.path.join(save_path, "validation_metrics.json"), "w") as json_file: 
    json.dump(metrics, json_file, indent=4)
    
# Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true=val_df["true"], y_pred=val_df["pred"], labels=val_df["true"].unique())
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_df["true"].unique())
disp.plot(ax = ax, xticks_rotation=30, cmap="Blues")
disp.figure_.savefig(os.path.join(save_path, "confusion_matrix.png"))

# Pycm 
from pycm import ConfusionMatrix
cm = ConfusionMatrix(val_df["true"].tolist(), val_df["pred"].tolist())
cm.save_html(os.path.join(save_path, "report"))
