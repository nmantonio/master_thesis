"""
python train.py --model_name densenet --fold_idx 1 --task detection --save_path /home/anadal/Experiments/TFM/master_thesis/test --epochs 15

python train.py --model_name --fold_idx --task --save_path --trainable_core --batch_size --augmentation_prob --epochs --patience --lr --database_path
"""


import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import argparse

from paths import *
import pandas as pd
import numpy as np

from models.utils import get_model, get_preprocessing_func
from models.tops import *

from keras import backend as K 
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.losses import CategoricalFocalCrossentropy
from utils import compute_loss_weights  

from data_loader import get_dataset
from utils import compute_graphics
import json

parser = argparse

parser = argparse.ArgumentParser(description='Script for training a model')

parser.add_argument('--model_name', type=str, default='xception', help='Name of the model (default: xception)', choices=['xception', 'mobilenet', 'densenet'])
parser.add_argument('--fold_idx', type=str, required=True, help='Fold index')
parser.add_argument('--task', type=str, required=True, help='Type of task', choices=['detection', 'classification'])
parser.add_argument('--save_path', type=str, required=True, help='Path to save the trained model')
parser.add_argument('--trainable_core', type=bool, default=True, help='Whether to train the core of the model (only if pretrained=True) (default: True)')
parser.add_argument('--pretrained', type=bool, default=True, help='Whether to use pretrained weights (ImageNet) (default: True)')
parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer to be used during training (default: "adam")', choices=["adam", "sgd"])
parser.add_argument('--epochs', type=int, default=300, help='Epochs for training (default: 300)')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training (default: 0.001)')
parser.add_argument('--patience', type=int, default=10, help='Patience to perform earlystopping (default: 10)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
parser.add_argument('--augmentation_prob', type=float, default=0, help='Probability of augmentation (default: 0)')
parser.add_argument('--database', type=str, default="noncropped", help='Database to be used (default: "noncropped")', choices=["cropped", "noncropped"])
parser.add_argument('--top_idx', type=int, default=1, help='Top index (see tops.py) (default=1)')
parser.add_argument('--loss', type=str, default="categorical_crossentropy", help='Loss function to be used (default: categorical_crossentropy)', choices=["categorical_crossentropy", "categorical_focal_crossentropy"])



args = parser.parse_args()

# Assigning argparse values to variables
model_name = args.model_name
trainable_core = args.trainable_core
batch_size = args.batch_size
fold_idx = args.fold_idx
task = args.task
augmentation_prob = args.augmentation_prob
epochs = args.epochs
patience = args.patience
lr = args.lr
pretrained = args.pretrained
top_idx = args.top_idx
loss = args.loss
optimizer = args.optimizer

if args.database == "cropped":
    database = CROPPED_DATABASE_PATH
else: # noncropped
    database = DATABASE_PATH


save_path = r"{}".format(args.save_path)
os.makedirs(save_path, exist_ok=True)
with open(os.path.join(save_path, "train_args.json"), "w") as json_file: 
    json.dump(args.__dict__, json_file, indent=4)


# Preprocessing
preprocessing_func = get_preprocessing_func(name=model_name, pretrained=pretrained, task=task, database=database)

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
    database_path=database,
    fold_idx=fold_idx, 
    batch_size=batch_size,
    mode="train", # train or val/test, 
    task=task, # detection or classification
    repeat=True,
    preprocessing_func = preprocessing_func,
    augmentation=augmentation_prob
)

val_dataset = get_dataset(
    database_path=database,
    fold_idx=fold_idx, 
    batch_size=batch_size,
    mode="val", # train or val/test, 
    task=task, # detection or classification
    repeat=True,
    preprocessing_func = preprocessing_func,
    augmentation=0
)

# base_model = Xception(weights=None, include_top=False, input_shape=(512, 512, 1))
base_model = get_model(name=model_name, pretrained=pretrained, trainable=trainable_core)

x = base_model.output
x = get_top(top_idx=top_idx)(x)
predictions = Dense(n_classes, activation="softmax")(x)

# Create the fine-tuned model
model = Model(inputs=base_model.input, outputs=predictions, name=model_name)
model.summary(line_length=175)

if loss == "categorical_focal_crossentropy":
    weights = compute_loss_weights(task, df) 
    loss = CategoricalFocalCrossentropy(alpha=weights)
    
if optimizer == "adam":
    optimizer = Adam(learning_rate=lr)
elif optimizer == "sgd": 
    optimizer = SGD(learning_rate=lr, momentum=0.5)


model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

model.fit(
    train_dataset,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=val_steps_per_epoch,
    callbacks = [
        EarlyStopping(patience=patience, restore_best_weights=True, verbose=1, start_from_epoch=10),
        CSVLogger(os.path.join(save_path, "train_log.csv")), 
        # ReduceLROnPlateau(
        #     monitor="val_loss", 
        #     factor=0.1, 
        #     patience=5, 
        #     verbose=1, 
        #     min_lr=0.000001
        # )
    ],
    verbose=2
)

model.save(os.path.join(save_path, "model.keras"))

compute_graphics(
    train_log = os.path.join(save_path, "train_log.csv"), 
    save_path = save_path
)

K.clear_session()
