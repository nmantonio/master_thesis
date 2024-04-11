import pickle
import numpy as np
import pandas as pd
from paths import *
import cv2

def load_encoder(encoder_path):
    with open(encoder_path, "rb") as encoder_file: 
        encoder = pickle.load(encoder_file)
        
    return encoder


def compute_graphics(train_log, save_path):
    import pandas as pd
    import os
    from matplotlib import pyplot as plt    

    df = pd.read_csv(train_log)

    # Extract data from the DataFrame
    epochs = df['epoch']
    accuracy = df['accuracy']
    loss = df['loss']
    val_accuracy = df['val_accuracy']
    val_loss = df['val_loss']

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'accuracy_plot.png'))
    # plt.show()

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_plot.png'))
    # plt.show()
    
def compute_loss_weights(task, df):
    n = len(df)
    weights = np.zeros(shape=(len(df.classification.unique())))
    if task == "classification":
        from paths import CLASSIFICATION_ENCODER
        encoder = load_encoder(CLASSIFICATION_ENCODER)
        
        for disease in df.classification.unique():
            disease_idx  = np.argmax(encoder.transform(pd.DataFrame({task: [disease]})).toarray())

            weights[disease_idx] = (n - len(df[df["classification"] == disease])) / n
            
        weights = (weights - np.min(weights)/2) / (np.max(weights) - np.min(weights)/2)
    # print(weights)
    
    # from sklearn.utils.class_weight import compute_class_weight
    # encoded = df[task]
    # weights = compute_class_weight(class_weight="balanced", y=encoded, classes=np.unique(encoded))
    
    return weights

def compute_mu_sigma(fold_idx, database_path, task):
    task = task.lower()
    if task == "detection":
        df = pd.read_csv(DETECTION_CSV)
    elif task == "classification":
        df = pd.read_csv(CLASSIFICATION_CSV)
    else: 
        raise ValueError(f"Task not supported. Try 'detection' or 'classification'.")
    
    FOLDS = ["1", "2", "3", "4", "5"]
    
    FOLDS.remove(fold_idx)
    df = df[df["split"].isin(FOLDS)]

    sum_pixel_values = 0
    sum_sq_pixel_values = 0
    total_pixels = 0
    
    for idx, row in df.iterrows():
        image = cv2.imread(os.path.join(database_path, row["new_filename"]), 0)
        image = (image).astype(np.float32) / 255.0

        if database_path == CROPPED_DATABASE_PATH:
            mask = cv2.imread(os.path.join(MASKS_PATH, row["new_filename"]), 0)
            mask = cv2.resize(mask, (512, 512))
            image = image[mask>0]
        
        total_pixels += image.size
        sum_pixel_values += np.sum(image)
        sum_sq_pixel_values += np.sum(np.square(image))
    
    mean = sum_pixel_values / (total_pixels)
    std = np.sqrt((sum_sq_pixel_values / (total_pixels)) - np.square(mean))
    
    return mean, std

# print("Detection") 
# for fold_idx in range(1, 6):
#     mean, std = compute_mu_sigma(str(fold_idx), DATABASE_PATH, "detection")
#     print(f"{fold_idx}: {mean}, {std}")

  
# for fold_idx in range(1, 6):
#     mean, std = compute_mu_sigma(str(fold_idx), CROPPED_DATABASE_PATH, "detection")
#     print(f"{fold_idx}: {mean}, {std}")
    
# print("Classification")
# for fold_idx in range(1, 6):
#     mean, std = compute_mu_sigma(str(fold_idx), DATABASE_PATH, "classification")
#     print(f"{fold_idx}: {mean}, {std}")


# for fold_idx in range(1, 6):
#     mean, std = compute_mu_sigma(str(fold_idx), CROPPED_DATABASE_PATH, "classification")
#     print(f"{fold_idx}: {mean}, {std}")
