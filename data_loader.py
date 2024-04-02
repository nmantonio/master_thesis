import cv2
import os
import numpy as np
import pandas as pd
from utils import load_encoder

def preprocess(image):
    return (image.astype(np.float32) / 127.5) - 1.0

def get_dataset(
    database_path,
    csv_path, 
    fold_idx, 
    batch_size,
    mode, 
    repeat,
    encoder_path,
    augmentation=0.25 # augment prob
):
    label_mode = "pathology" if "classification" in csv_path else "detection"
    print(f"Selected label mode: {label_mode}")
    encoder = load_encoder(encoder_path)
    
    FOLDS = ["1", "2", "3", "4", "5"]
    df = pd.read_csv(csv_path)
    if mode == "train":
        FOLDS.remove(fold_idx)
        df = df[df["split"].isin(FOLDS)]
        
        print("Training folds: ", df.split.unique())
                
    elif mode == "test":
        df = df[df["split"] == fold_idx]
        print("Validation fold: ", df.split.unique())
        
    else:
        raise ValueError(f"Invalid mode {mode}. Only available 'train' or 'test'.")
    
    data = list(zip(df["new_filename"], df[label_mode]))
    total_images = len(data)
    print(f"Total images per epoch ({mode}): {total_images}")   
     
    aug_idx = 0
    while True:
        np.random.shuffle(data)
        
        for start_idx in range(0, total_images, batch_size):
            end_idx = start_idx + batch_size
            if end_idx > total_images:
                end_idx = total_images

            image_batch = np.zeros(shape=((end_idx - start_idx,) + (512, 512, 3)))

            label_batch = np.zeros(shape=((end_idx - start_idx, len(encoder.get_feature_names_out()))))
            print(label_batch.shape)


            
            for idx, (filename, disease) in enumerate(data[start_idx:end_idx]):
                image = cv2.imread(os.path.join(database_path, filename))
                
                if augmentation > 0: 
                    aug = ""
                    if np.random.rand() < augmentation:
                        gamma = np.random.uniform(0.8, 1.2)
                        image = np.power(image / 255.0, gamma)
                        image = (image * 255).astype(np.uint8)
                        
                        aug += "gamma"
                        
                    # if np.random.rand() < augmentation:
                    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    #     image = clahe.apply(image)
                        
                        # aug += "clahe"
                        
                    if np.random.rand() < augmentation:
                        image = 255 - image
                        
                        aug += "inv"
                    
                    if np.random.rand() < augmentation:
                        angle = np.random.randint(-15, 15)  
                        rows, cols = image.shape[:2]
                        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                        image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
                        
                        aug += "rot"
                        
                    if np.random.rand() < 0.1:    
                        cv2.imwrite(os.path.join(r"/home/anadal/Experiments/TFM/AUGMENTATED_IMAGES", f"{aug}_{aug_idx}.png"), image)
                    aug_idx += 1
                
                # image = np.expand_dims(image, axis=-1)

                # cv2.imshow(filename, image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # image_batch[idx] = (image / 127.5 - 1).astype(np.float32)
                image_batch[idx] = preprocess(image)

                label_batch[idx] = encoder.transform(pd.DataFrame({"pathology": [disease]})).toarray()
                print(label_batch)

            yield (image_batch, label_batch)

        if not repeat:
            break


if __name__ == "__main__":
    TFM_PATH = r"/home/anadal/Experiments/TFM"

    DATABASE_PATH = r"/home/anadal/Experiments/TFM/PROCESSED_DATABASE"

    DETECTION_CSV = r"/home/anadal/Experiments/TFM/master_thesis/abnormal_detection_selection.csv"
    CLASSIFICATION_CSV = r"/home/anadal/Experiments/TFM/master_thesis/disease_classification_selection.csv"
    
    DETECTION_ENCODER = r"/home/anadal/Experiments/TFM/master_thesis/detection_encoder"
    CLASSIFICATION_ENCODER = r"/home/anadal/Experiments/TFM/master_thesis/classification_encoder"

    dataset = get_dataset(
        DATABASE_PATH,
        CLASSIFICATION_CSV, 
        fold_idx="1",
        batch_size=6, 
        mode="train",
        repeat=True, 
        encoder_path = CLASSIFICATION_ENCODER
    )
    
    next(dataset)