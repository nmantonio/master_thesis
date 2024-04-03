import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from paths import *
import time

"""
python train.py --model_name --fold_idx --task --save_path --trainable_core --batch_size --augmentation_prob --epochs
"""

train_name = "mobilenet_test_trainable_false"
save_path = os.path.join(TRAIN_PATH, train_name)
os.makedirs(save_path)

model_name = "mobilenet"
task = "detection"
trainable_core = False
epochs = 5
batch_size = 32
augmentation_prob = 0


FOLDS = ["1", "2", "3", "4", "5"]

for fold_idx in FOLDS: 
    
    print(f"\nStarting fold {fold_idx} training!")
    
    instruction = f"python train.py \
        --model_name {model_name}\
            --fold_idx {fold_idx}\
                --task {task}\
                    --save_path {os.path.join(save_path, fold_idx)}\
                        --trainable_core {trainable_core}\
                            --epochs {epochs}\
                                --batch_size {batch_size}\
                                    --augmentation_prob {augmentation_prob}\
                                        | tee {os.path.join(save_path, f'log_{fold_idx}.txt')}"
    
    os.system(instruction)
    
    print(f"\nFold {fold_idx} training finished!")
    print("Sleeping 30 secs!")
    time.sleep(1)
    