import cv2
import os
import numpy as np
import pandas as pd

DETECTION_ENCODING = {
    "normal": np.array([1, 0]), 
    "bacterialpneumonia": np.array([0, 1]), 
    "viralpneumonia": np.array([0, 1]), 
    "lungopacity": np.array([0, 1]), 
    "pulmonaryfibrosis": np.array([0, 1]), 
    "tuberculosis": np.array([0, 1]), 
    "covid": np.array([0, 1])
}

CLASSIFICATION_ENCODING = {
    "normal": np.array([1, 0, 0, 0, 0, 0, 0]), 
    "bacterialpneumonia": np.array([0, 1, 0, 0, 0, 0, 0]), 
    "viralpneumonia": np.array([0, 0, 1, 0, 0, 0, 0]), 
    "lungopacity": np.array([0, 0, 0, 1, 0, 0, 0]), 
    "pulmonaryfibrosis": np.array([0, 0, 0, 0, 1, 0, 0]), 
    "tuberculosis": np.array([0, 0, 0, 0, 0, 1, 0]),
    "covid": np.array([0, 0, 0, 0, 0, 0, 1]) 
}

def get_pathology_label(filename, mode="detection"):
    pathology = filename.split("_")[0]
    
    if mode.lower()=="detection":
        return DETECTION_ENCODING[pathology]
    elif mode.lower()=="classification":
        return CLASSIFICATION_ENCODING[pathology]
    else: 
        raise ValueError(f"{mode} mode not available! Try 'detection' or 'classification'")

def decode_prediction(prediction, mode="detection"):
    if mode.lower() == "detection":
        labels = ["normal", "abnormal"]
        label_index = np.argmax(prediction)
        return labels[label_index]
    elif mode.lower() == "classification":
        labels = CLASSIFICATION_ENCODING.keys()
        pathology_index = np.argmax(prediction)
        return labels[pathology_index]
    else:
        raise ValueError(f"Invalid mode {mode}. Only available 'detection' or 'classification'")


def preprocess(image):
    return (image.astype(np.float32) / 127.5) - 1.0

def get_dataset(
    database_path,
    csv_path, 
    fold_idx, 
    batch_size,
    mode, 
    repeat
):
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
    
    filenames = list(df["new_filename"])
    total_images = len(filenames)
    print(f"Total images per epoch ({mode}): {total_images}")    
    while True:
        np.random.shuffle(filenames)
        
        for start_idx in range(0, total_images, batch_size):
            end_idx = start_idx + batch_size
            if end_idx > total_images:
                end_idx = total_images

            image_batch = np.zeros(shape=((end_idx - start_idx,) + (512, 512, 1)))
            label_batch = np.zeros(shape=((end_idx - start_idx, 7)))
            
            for idx, filename in enumerate(filenames[start_idx:end_idx]):
                image = cv2.imread(os.path.join(database_path, filename), 0)
                image = np.expand_dims(image, axis=-1)

                # cv2.imshow(filename, image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # image_batch[idx] = (image / 127.5 - 1).astype(np.float32)
                image_batch[idx] = preprocess(image)

                label_batch[idx] = get_pathology_label(filename, mode="classification")
                # print(label_batch)

            yield (image_batch, label_batch)

        if not repeat:
            break


if __name__ == "__main__":
    DATABASE_PATH = r"/home/marc/ANTONIO_EXPERIMENTS/TFM/PROCESSED_DATABASE"
    TFM_PATH = r"/home/marc/ANTONIO_EXPERIMENTS/TFM"

    ABNORMAL_CSV = r"/home/marc/ANTONIO_EXPERIMENTS/TFM/master_thesis/abnormal_detection_selection.csv"
    DISEASE_CSV = r"/home/marc/ANTONIO_EXPERIMENTS/TFM/master_thesis/disease_classification_selection.csv"


    dataset = get_dataset(
        DATABASE_PATH,
        DISEASE_CSV, 
        fold_idx="1",
        batch_size=6, 
        mode="train",
        repeat=True
    )
    
    next(dataset)