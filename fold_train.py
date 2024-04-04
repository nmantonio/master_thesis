import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from paths import *
import shutil
import time

"""
python train.py --model_name --fold_idx --task --save_path --trainable_core --batch_size --augmentation_prob --epochs --patience --lr --pretrained
"""

train_name = "xception_test_cropped_augment_detection"
save_path = os.path.join(TRAIN_PATH, train_name)
if os.path.exists(save_path):
    response = input("The directory already exists. Do you want to remove it? (y/N): ")
    if response.lower().strip() == 'y':
        shutil.rmtree(save_path)
    else:
        print("Exiting program...")
        exit()
os.makedirs(save_path)

model_name = "xception"
task = "detection"
trainable_core = True
pretrained = False
epochs = 100
lr = 0.001
patience = 10
batch_size = 64
augmentation_prob = 0.25
database = CROPPED_DATABASE_PATH
top_idx = 1


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
        --epochs {epochs}\
        --lr {lr}\
        --patience {patience}\
        --batch_size {batch_size}\
        --augmentation_prob {augmentation_prob}\
        --database_path {database}\
        --top_idx {top_idx}\
        | tee {os.path.join(save_path, f'log_{fold_idx}.txt')}"
    
    os.system(instruction)
    
    print(f"\nFold {fold_idx} training finished!")
    print("\nSleeping 30 secs!")
    time.sleep(30)
    