import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import paths
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import Xception, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.optimizers.schedules import ExponentialDecay
# from metrics import F1Score
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_loader import get_dataset

DATABASE_PATH = r"/home/anadal/Experiments/TFM/PROCESSED_DATABASE"
TFM_PATH = r"/home/anadal/Experiments/TFM"

DETECTION_CSV = r"/home/anadal/Experiments/TFM/master_thesis/abnormal_detection_selection.csv"
# CLASSIFICATION_CSV = r"/home/anadal/Experiments/TFM/master_thesis/disease_classification_selection.csv"

try_name = "Xception_adam_0.00001_abnormal_transfer"
SAVE = os.path.join(r"/home/anadal/Experiments/TFM/master_thesis/Xception", try_name)
os.makedirs(SAVE, exist_ok=True)

df = pd.read_csv(DETECTION_CSV)
batch_size = 32
FOLDS = ["2", "3", "4", "5"]
fold_idx = "1"

steps_per_epoch = np.ceil(df[df["split"].isin(FOLDS)].shape[0] / batch_size)
val_steps_per_epoch = np.ceil(df[df["split"] == fold_idx].shape[0] / batch_size)
print("STEPS PER EPOCH: ", steps_per_epoch)
print("VAL STEPS PER EPOCH: ", val_steps_per_epoch)

train_dataset = get_dataset(
    database_path=DATABASE_PATH,
    csv_path = DETECTION_CSV, 
    fold_idx=fold_idx,
    batch_size=batch_size, 
    mode="train",
    repeat=True, 
    augmentation=0
)

val_dataset = get_dataset(
    database_path=DATABASE_PATH,
    csv_path = DETECTION_CSV, 
    fold_idx=fold_idx,
    batch_size=batch_size, 
    mode="test",
    repeat=True, 
    augmentation=0
)

base_model = Xception(weights="imagenet", include_top=False, input_shape=(512, 512, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)

predictions = Dense(2, activation='softmax')(x)

# Create the fine-tuned model
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()

model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_dataset, epochs=400, steps_per_epoch=steps_per_epoch, validation_data=val_dataset, validation_steps=val_steps_per_epoch, callbacks = [EarlyStopping(patience=30, restore_best_weights=True), CSVLogger(os.path.join(SAVE, "train_log.csv"))])

model.save(os.path.join(SAVE, "model.h5"))