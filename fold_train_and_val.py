from paths import *

"""
python fold_train.py --train_name xception_classification_pretrained_nontrainable --model_name xception --task classification --trainable_core False --pretrained True --epochs 100 --lr 0.0001 --patience 10 --batch_size 32 --augmentation_prob 0.25 --database cropped --top_idx 1 --loss categorical_focal_crossentropy
"""

import argparse

parser = argparse.ArgumentParser(description="Scrip for performing a 5-fold training")
parser.add_argument("--device", type=str, default="0", help="GPU device (default: '0')")
parser.add_argument("--train_name", type=str, required=True, help="Name of the folder containing the trainings")
parser.add_argument("--model_name", type=str, default="xception", help="Name of the model", choices=['xception', 'mobilenet', 'densenet'])
parser.add_argument('--task', type=str, required=True, help='Type of task', choices=['detection', 'classification'])
parser.add_argument('--trainable_core', type=bool, default=True, help='Whether to train the core of the model (only if pretrained=True) (default: True)')
parser.add_argument("--pretrained", type=bool, default=True, help="Whether to use pretrained weights")
parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer to be used during training (default: "adam")', choices=["adam", "sgd"])
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument('--augmentation_prob', type=float, default=0, help='Probability of augmentation (default: 0)')
parser.add_argument('--database', type=str, default="noncropped", help='Database to be used (default: "noncropped")', choices=["cropped", "noncropped"])
parser.add_argument('--top_idx', type=int, default=1, help='Top index (see tops.py) (default=1)')
parser.add_argument('--loss', type=str, default="categorical_crossentropy", help='Loss function to be used (default: categorical_crossentropy)', choices=["categorical_crossentropy", "categorical_focal_crossentropy"])

args = parser.parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"]=args.device
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import shutil
import time

train_name = args.train_name
save_path = os.path.join(TRAIN_PATH, train_name)
if os.path.exists(save_path):
    response = input("The directory already exists. Do you want to remove it? (y/N): ")
    if response.lower().strip() == 'y':
        shutil.rmtree(save_path)
    else:
        print("Exiting program...")
        exit()
os.makedirs(save_path)

model_name = args.model_name
task = args.task
trainable_core = args.trainable_core
pretrained = args.pretrained
epochs = args.epochs
lr = args.lr
patience = args.patience
batch_size = args.batch_size
augmentation_prob = args.augmentation_prob
database = args.database
top_idx = args.top_idx
loss = args.loss
optimizer = args.optimizer


FOLDS = ["1", "2", "3", "4", "5"]

for fold_idx in FOLDS: 
    
    print(f"\nStarting fold {fold_idx} training!")
    
    instruction = f"python train.py \
        --model_name {model_name}\
        --fold_idx {fold_idx}\
        --task {task}\
        --save_path {os.path.join(save_path, fold_idx)}\
        --trainable_core {trainable_core}\
        --pretrained {pretrained}\
        --optimizer {optimizer}\
        --epochs {epochs}\
        --lr {lr}\
        --patience {patience}\
        --batch_size {batch_size}\
        --augmentation_prob {augmentation_prob}\
        --database {database}\
        --top_idx {top_idx}\
        --loss {loss}\
        | tee {os.path.join(save_path, f'log_{fold_idx}.txt')}"
    
    os.system(instruction)
    
    print(f"\nFold {fold_idx} training finished!")
    
    print(f"\nStarting fold {fold_idx} validation.")
    
    # Compute fold validation files
    val_instruction = f"python validation.py --train_path {os.path.join(save_path, fold_idx)}"
    os.system(val_instruction)
    
    print(f"\nFold {fold_idx} validation finished!.")
    
    print("\nSleeping 30 secs!")
    time.sleep(1)
    
# Store mean val results
import json
import pandas as pd

global_metrics = {}
for fold_idx in FOLDS:
    fold_path = os.path.join(save_path, fold_idx)
    
    with open(os.path.join(fold_path, "val", "validation_metrics.json"), 'r') as json_file:
        data = json.load(json_file)
    
    # Dict variables initialization to zero
    if len(global_metrics.keys()) == 0:
        for key in data.keys():
            global_metrics[key] = 0
            
    for key, value in data.items():
        global_metrics[key] += data[key]
        

for key in global_metrics.keys():
    global_metrics[key] /= len(FOLDS)
    
with open(os.path.join(save_path, "avg_results.json"), "w") as results_file:
    json.dump(global_metrics, results_file, indent=4)
    
results_df = pd.DataFrame(data=global_metrics, index=[0])
results_df.to_excel(os.path.join(save_path, "avg_results.xlsx"), index=False, header=True)
    