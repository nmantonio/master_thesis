import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from natsort import natsorted
import cv2
import numpy as np

db_path = r"/home/anadal/Experiments/TFM/PROCESSED_DATABASE"

model_path = r"/home/anadal/Experiments/TFM/master_thesis/MobileNetV2/Mobilenet_adam_0.0001_abnormal/model.h5"

fold_idx = "1"

df = r"/home/anadal/Experiments/TFM/master_thesis/abnormal_detection_selection.csv"


model = load_model(model_path)

df = pd.read_csv(df)
df = df[df["split"] == str(fold_idx)]
print(df)

val_df = pd.DataFrame(columns=["filename", "prediction"])

for idx, filename in enumerate(natsorted(df["new_filename"])):
    image = cv2.imread(os.path.join(db_path, filename), 0)
    image = (image.astype(np.float32) / 127.5) - 1.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    
    pred = model.predict(image)
    pred = np.argmax(pred)
    
    new_row = {"filename": filename, "prediction": pred} 
    val_df = pd.concat([val_df, pd.DataFrame([new_row])], ignore_index=True)
    
        
    if idx < 5:
        print(val_df)
    
val_df.to_csv(r"/home/anadal/Experiments/TFM/master_thesis/MobileNetV2/Mobilenet_adam_0.0001_abnormal/predictions.csv", index=False)


