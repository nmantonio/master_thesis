import cv2
import os
import numpy as np
import pandas as pd
from utils import load_encoder
from paths import *

def get_dataset(
    database_path,
    fold_idx, 
    batch_size,
    mode, # train or val/test, 
    task, # detection or classification
    preprocessing_func,
    repeat=True,
    augmentation=0.25 # augment prob
):
    fold_idx = str(fold_idx)
    task = task.lower()
    if task == "detection":
        encoder = load_encoder(DETECTION_ENCODER)
        df = pd.read_csv(DETECTION_CSV)
    elif task == "classification":
        encoder = load_encoder(CLASSIFICATION_ENCODER)
        df = pd.read_csv(CLASSIFICATION_CSV)
    elif task == "abnormal_classification":
        encoder = load_encoder(ABNORMAL_CLASSIFICATION_ENCODER)
        df = pd.read_csv(CLASSIFICATION_CSV)
        df = df[df["classification"] != "normal"]
        task = "classification"
    else: 
        raise ValueError(f"Task not supported. Try 'detection' or 'classification'.")
    
    FOLDS = ["1", "2", "3", "4", "5"]
    
    if mode == "train":
        FOLDS.remove(fold_idx)
        df = df[df["split"].isin(FOLDS)]
        print("\nTraining folds: ", df.split.unique())
                
    elif (mode == "val"):
        df = df[df["split"] == fold_idx]
        print("\nValidation fold: ", df.split.unique())
        
    elif (mode == "test"):
        df = df[df["split"] == "test"]
        print("\nTesting on: ", df.split.unique())
    else:
        raise ValueError(f"Invalid mode {mode}. Only available 'train', 'val' or 'test'.")
    
    data = list(zip(df["new_filename"], df[task]))
    total_images = len(data)
    print(f"\nTotal images per epoch ({mode}): {total_images}")   
     
    aug_idx = 0
    while True:
        np.random.shuffle(data)
        
        for start_idx in range(0, total_images, batch_size):
            end_idx = start_idx + batch_size
            if end_idx > total_images:
                end_idx = total_images

            image_batch = np.zeros(shape=((end_idx - start_idx,) + (512, 512, 1)))

            label_batch = np.zeros(shape=((end_idx - start_idx, len(encoder.get_feature_names_out()))))
            
            for idx, (filename, disease) in enumerate(data[start_idx:end_idx]):
                image = cv2.imread(os.path.join(database_path, filename), 0)
                if database_path == CROPPED_DATABASE_PATH: 
                    mask = cv2.imread(os.path.join(MASKS_PATH, filename), 0)
                    mask = cv2.resize(mask, (512, 512))
                
                if (augmentation > 0) and (mode == "train"): 
                    aug = ""
                    if np.random.rand() < augmentation:
                        gamma = np.random.uniform(0.8, 1.2)
                        image = np.power(image / 255.0, gamma)
                        image = (image * 255).astype(np.uint8)
                        
                        aug += "gamma"
                        
                    if np.random.rand() < augmentation:
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        image = clahe.apply(image)
                        
                        aug += "clahe"
                        
                    if np.random.rand() < augmentation:
                        image = 255 - image
                        
                        aug += "inv"
                    
                    if np.random.rand() < augmentation:
                        angle = np.random.randint(-15, 15)  
                        rows, cols = image.shape[:2]
                        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                        image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
                        
                        if database_path == CROPPED_DATABASE_PATH: 
                            mask =  cv2.warpAffine(mask, rotation_matrix, (cols, rows))
                        
                        aug += "rot"
                        
                    if (np.random.rand() < 0.005) and os.path.exists(AUGMENTED_IMAGES_CHECK):    
                        cv2.imwrite(os.path.join(AUGMENTED_IMAGES_CHECK, f"{aug}_{aug_idx}.png"), image)
                    aug_idx += 1
                
                # cv2.imwrite(os.path.join(IMAGES_CHECK, filename), image)
                
                if database_path == CROPPED_DATABASE_PATH:
                    image = cv2.bitwise_and(image, image, mask=mask)
                    image = np.expand_dims(image, axis=-1)
                    image_batch[idx] = preprocessing_func(image, mask=mask)
                else: 
                    image = np.expand_dims(image, axis=-1)
                    image_batch[idx] = preprocessing_func(image)

                label_batch[idx] = encoder.transform(pd.DataFrame({task: [disease]})).toarray()
                # print(label_batch[idx])

            yield (image_batch, label_batch)

        if not repeat:
            break


# if __name__ == "__main__":

#     dataset = get_dataset(
#         database_path = DATABASE_PATH,
#         fold_idx = 1, 
#         batch_size = 8,
#         mode = "train", # train or val/test, 
#         task = "detection", # detection or classification
#     )
    
#     next(dataset)