import pandas as pd
from natsort import natsorted
import json
import cv2

from paths import *

if __name__ == "__main__":
    
    # train_list = natsorted(os.listdir(TRAIN_PATH))
    train_list = ["9_model_name_xception_task_detection_pretrained_True_trainable_core_True_optimizer_sgd_epochs_300_lr_0.001_patience_5_batch_size_64_augmentation_prob_0.0_database_cropped_top_idx_1_loss_categorical_crossentropy"]


    for train_folder in train_list: 
        # print(f"Starting gradcam: {train_folder}")
        for fold_idx in range(1, 2):
            fold_path = os.path.join(TRAIN_PATH, train_folder, str(fold_idx))
            print(f"Fold idx: {fold_idx}")
            if os.path.exists(os.path.join(fold_path, "bbox_gradcam")):
                continue
            os.makedirs(os.path.join(fold_path, "bbox_gradcam"), exist_ok=True)
            
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
                
            # preprocess = get_preprocessing_func(name=model_name, pretrained=pretrained, task=task, database=database)
              
            if task == "detection": 
                # encoder = load_encoder(DETECTION_ENCODER)
                df = pd.read_csv(DETECTION_CSV)
            elif task == "classification":
                # encoder = load_encoder(CLASSIFICATION_ENCODER)
                df = pd.read_csv(CLASSIFICATION_CSV)
            else: 
                raise ValueError(f"Task not supported. Try 'detection' or 'classification'.")

            df = df[df["split"] == str(fold_idx)]
            df = df[df["classification"] == "pulmonary_fibrosis"]
            df = df[df["database"] == "DS6"]
            
            df_bbox = pd.read_csv(BBOX_CSV)
            df_bbox = df_bbox[df_bbox["class_id"] == 13]
            # df_bbox = df_bbox[df_bbox["image_id"].isin(names)]

            for idx, row in df.iterrows():
                image = cv2.imread(os.path.join(fold_path, "gradcam", row["new_filename"]))
                name = ".".join(row["original_filename"].split('.')[:-1])
                aux = df_bbox[df_bbox["image_id"] == name]
                
                shape = (df.loc[idx, "shape1"], df.loc[idx, "shape2"])
                for idx_2, row_2 in aux.iterrows():
                    x_min, y_min = int((row_2["x_min"]*512) // shape[1]), int((row_2["y_min"]*512) // shape[0])
                    x_max, y_max = int((row_2["x_max"]*512) // shape[1]), int((row_2["y_max"]*512) // shape[0])
                    print(x_min, y_min, x_max, y_max)
                    
                    color = (0, 0, 255)  # BGR color format, here it's green
                    thickness = 1  # Thickness of the rectangle border      
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
                    
                    # label_text = row_2["rad_id"]
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # font_scale = 1
                    # font_thickness = 2
                    # text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
                    # text_x = x_min + (x_max - x_min - text_size[0]) // 2
                    # text_y = y_min + (y_max - y_min + text_size[1]) // 2
                    # cv2.putText(image, label_text, (text_x, text_y), font, font_scale, color, font_thickness)
                cv2.imwrite(os.path.join(fold_path, "bbox_gradcam", row["new_filename"]), image)
                
                # print(aux)
                # raise Exception
