import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import paths
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
# from metrics import F1Score
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_loader import get_dataset

DATABASE_PATH = r"C:\Users\tonin\Desktop\Master\TFM\PROCESSED_DATABASE"
TFM_PATH = r"C:\Users\tonin\Desktop\Master\TFM"

ABNORMAL_CSV = r"C:\Users\tonin\Desktop\Master\TFM\master_thesis\abnormal_detection_selection.csv"
DISEASE_CSV = r"C:\Users\tonin\Desktop\Master\TFM\master_thesis\disease_classification_selection.csv"

df = pd.read_csv(ABNORMAL_CSV)
batch_size = 7
FOLDS = ["2", "3", "4", "5"]
fold_idx = "1"

steps_per_epoch = np.ceil(df[df["split"].isin(FOLDS)].shape[0] / batch_size)
val_steps_per_epoch = np.ceil(df[df["split"] == fold_idx].shape[0] / batch_size)

train_dataset = get_dataset(
    database_path=DATABASE_PATH,
    csv_path = ABNORMAL_CSV, 
    fold_idx=fold_idx,
    batch_size=batch_size, 
    mode="train",
    repeat=True
)

val_dataset = get_dataset(
    database_path=DATABASE_PATH,
    csv_path = ABNORMAL_CSV, 
    fold_idx=fold_idx,
    batch_size=batch_size, 
    mode="test",
    repeat=True
)

base_model = Xception(weights=None, include_top=False, input_shape=(512, 512, 1))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)

predictions = Dense(2, activation='softmax')(x)

# Create the fine-tuned model
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()

model.compile(optimizer=SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_dataset, epochs=300, steps_per_epoch=steps_per_epoch, validation_data=val_dataset, validation_steps=val_steps_per_epoch, callbacks = [EarlyStopping(patience=10, restore_best_weights=True)])

model.save(os.path.join(TFM_PATH, "SGD_lr001_categorical.h5"))