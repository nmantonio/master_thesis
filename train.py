"""
python train.py --model_name mobilenet --fold_idx 1 --task detection --save_path /home/anadal/Experiments/TFM/master_thesis/mobilenet_train --epochs 15

python train.py --model_name --fold_idx --task --save_path --trainable_core --batch_size --augmentation_prob --epochs
"""


import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import argparse

from paths import *
import pandas as pd
import numpy as np

from models.utils import modify_model, get_preprocessing_func
from models.tops import *

from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, CSVLogger

from data_loader import get_dataset
from utils import compute_graphics

parser = argparse

parser = argparse.ArgumentParser(description='Script for training a model')

parser.add_argument('--model_name', type=str, default='xception', help='Name of the model (default: xception)', choices=['xception', 'mobilenet'])
parser.add_argument('--fold_idx', type=str, required=True, help='Fold index')
parser.add_argument('--task', type=str, required=True, help='Type of task', choices=['detection', 'classification'])
parser.add_argument('--save_path', type=str, required=True, help='Path to save the trained model')
parser.add_argument('--trainable_core', type=bool, default=True, help='Whether to train the core of the model (default: True)')
parser.add_argument('--epochs', type=int, default=300, help='Epochs for training (default: 300)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
parser.add_argument('--augmentation_prob', type=float, default=0, help='Probability of augmentation (default: 0)')


args = parser.parse_args()

# Assigning argparse values to variables
model_name = args.model_name
trainable_core = args.trainable_core
batch_size = args.batch_size
fold_idx = args.fold_idx
task = args.task
augmentation_prob = args.augmentation_prob
epochs = args.epochs

save_path = r"{}".format(args.save_path)
os.makedirs(save_path)

# Preprocessing
preprocessing_func = get_preprocessing_func(name=model_name)

# Read CSV
if task == "detection":
    df = pd.read_csv(DETECTION_CSV)
    n_classes = 2
elif task == "classification":
    df = pd.read_csv(CLASSIFICATION_CSV)
    n_classes = 7
else: 
    raise ValueError(f"{task} not available! Try 'detection' or 'classification'.")

FOLDS = ["1", "2", "3", "4", "5"]
# FOLDS = ["1", "2"]
FOLDS.remove(str(fold_idx))

steps_per_epoch = np.ceil(df[df["split"].isin(FOLDS)].shape[0] / batch_size)
val_steps_per_epoch = np.ceil(df[df["split"] == fold_idx].shape[0] / batch_size)

print("STEPS PER EPOCH: ", steps_per_epoch)
print("VAL STEPS PER EPOCH: ", val_steps_per_epoch)

train_dataset = get_dataset(
    database_path=DATABASE_PATH,
    fold_idx=fold_idx, 
    batch_size=batch_size,
    mode="train", # train or val/test, 
    task=task, # detection or classification
    repeat=True,
    preprocessing_func = preprocessing_func,
    augmentation=augmentation_prob
)

val_dataset = get_dataset(
    database_path=DATABASE_PATH,
    fold_idx=fold_idx, 
    batch_size=batch_size,
    mode="val", # train or val/test, 
    task=task, # detection or classification
    repeat=True,
    preprocessing_func = preprocessing_func,
    augmentation=0
)

# base_model = Xception(weights=None, include_top=False, input_shape=(512, 512, 1))
base_model = modify_model(name="xception", trainable=trainable_core)

x = base_model.output
x = get_top(top_idx=1)(x)
predictions = Dense(n_classes, activation="softmax")(x)

# Create the fine-tuned model
model = Model(inputs=base_model.input, outputs=predictions, name=model_name)
# model.summary(line_length=175)

model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_dataset,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=val_steps_per_epoch,
    callbacks = [
        EarlyStopping(patience=30, restore_best_weights=True),
        CSVLogger(os.path.join(save_path, "train_log.csv"))
    ],
    verbose=2
)

model.save(os.path.join(save_path, "model.keras"))

compute_graphics(
    train_log = os.path.join(save_path, "train_log.csv"), 
    save_path = save_path
)