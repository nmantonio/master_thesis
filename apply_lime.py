import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from lime import lime_image
from keras.models import load_model
import json
import numpy as np
import pandas as pd
import cv2
from skimage.segmentation import mark_boundaries

from utils import load_encoder
from paths import *
from models.utils import get_preprocessing_func

def predict(image):
    image = image.astype(np.float32)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    if image.shape[-1] == 3:
        print(image.shape)
        image = np.array([cv2.cvtColor(image[idx], cv2.COLOR_BGR2GRAY) for idx in range(image.shape[0])])
    for idx in range(0, image.shape[0]):
        cv2.imwrite(fr"/home/anadal/Experiments/TFM/temp/image{idx}.png",image[idx]*255)
    return model.predict(image)


explainer = lime_image.LimeImageExplainer()


if __name__ == "__main__":
    # train_list = natsorted(os.listdir(TRAIN_PATH))
    train_list = ["0_model_name_xception_task_detection_pretrained_True_trainable_core_True_optimizer_sgd_epochs_300_lr_0.001_patience_5_batch_size_64_augmentation_prob_0.0_database_noncropped_top_idx_1_loss_categorical_crossentropy"]
    for train_folder in train_list: 
        print(f"Starting lime: {train_folder}")
        for fold_idx in range(1, 2):
            fold_path = os.path.join(TRAIN_PATH, train_folder, str(fold_idx))
            print(f"Fold idx: {fold_idx}")
            # if os.path.exists(os.path.join(fold_path, "lime")):
            #     continue
            os.makedirs(os.path.join(fold_path, "lime"), exist_ok=True)
            
            model = load_model(os.path.join(fold_path, "model.keras"), compile=False)
            
            # for layer in model.layers:
            #     print(layer.name)
            # raise Exception

            with open(os.path.join(fold_path, "train_args.json"), 'r') as json_file:
                train_dict = json.load(json_file)
                
            task = train_dict["task"]
            model_name = train_dict["model_name"]
            fold_idx = train_dict["fold_idx"]
            pretrained = train_dict["pretrained"]
            task = train_dict["task"]
            database = train_dict["database"]
            if database == "cropped":
                database = CROPPED_DATABASE_PATH
            else: 
                database = DATABASE_PATH

            preprocess = get_preprocessing_func(name=model_name, pretrained=pretrained, task=task, database=database)
              
            if task == "detection": 
                encoder = load_encoder(DETECTION_ENCODER)
                df = pd.read_csv(DETECTION_CSV)
            elif task == "classification":
                encoder = load_encoder(CLASSIFICATION_ENCODER)
                df = pd.read_csv(CLASSIFICATION_CSV)
            else: 
                raise ValueError(f"Task not supported. Try 'detection' or 'classification'.")

            df = df[df["split"] == str(fold_idx)]
            
            for idx, row in df.iterrows():
                # print(row["new_filename"])
                img = cv2.imread(os.path.join(database, row["new_filename"]), 0)
                img = np.expand_dims(img, axis=-1)
                
                if database == CROPPED_DATABASE_PATH: 
                    mask = cv2.imread(os.path.join(MASKS_PATH, row["new_filename"]), 0)
                    mask = cv2.resize(mask, (512, 512))
                    img = preprocess(img, mask=mask)
                else: 
                    img = preprocess(img)
                img = np.squeeze(img)   

                explanation = explainer.explain_instance(img, predict, top_labels=2, hide_color=0, num_samples=100)

                temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
                # ol_image, heatmap = show_imgwithheat(os.path.join(database, row["new_filename"]), heatmap)
                # ol_image = cv2.cvtColor(ol_image, cv2.COLOR_BGR2RGB)
                
                lime_image = cv2.cvtColor(mark_boundaries(temp / 2 + 0.5, mask)*255, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(fold_path, "lime", row["new_filename"]), lime_image)
                # print(ol_image.shape)
                # cv2.imwrite(os.path.join(IMAGES_CHECK, row["new_filename"]), heatmap)

            else: 
                print(f"Fold {fold_idx} done!")
                del model
                
        else: 
            print(f"Folder {train_folder} done!")