import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import json

from paths import *
import cv2
import pandas as pd
from utils import load_encoder
from models.utils import get_preprocessing_func
from keras.models import load_model

from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.gradcam import Gradcam, GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam

from lime import lime_image
explainer = lime_image.LimeImageExplainer()

from skimage.segmentation import mark_boundaries


def lime_predict(image, save_input_example=False):
    image = image.astype(np.float32)
    if image.shape[-1] == 3:
        image = np.array([cv2.cvtColor(image[idx], cv2.COLOR_BGR2GRAY) for idx in range(image.shape[0])])
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=-1)
    
    if save_input_example:
        for idx in range(0, image.shape[0]):
            cv2.imwrite(os.path.join(save_input_example, f"image{idx}.png"),image[idx]*255)
    return model.predict(image)



LAST_CONV_LAYERS = {
    "xception": "block14_sepconv2_act",
    "mobilenet": "multiply_17", 
    "densenet": "relu"
}

train_list = ["0_model_name_xception_task_detection_pretrained_True_trainable_core_True_optimizer_sgd_epochs_300_lr_0.001_patience_5_batch_size_64_augmentation_prob_0.0_database_noncropped_top_idx_1_loss_categorical_crossentropy"]
for train_folder in train_list: 
    print(f"Starting explainability: {train_folder}")
    for fold_idx in range(1, 2):
        fold_path = os.path.join(TRAIN_PATH, train_folder, str(fold_idx))
        print(f"Fold idx: {fold_idx}")
        
        figure_path = os.path.join(fold_path, "explainability_figures")
        # if os.path.exists(figure_path):
        #     continue
        os.makedirs(figure_path, exist_ok=True)
        os.makedirs(os.path.join(fold_path, "in"), exist_ok=True)
        os.makedirs(os.path.join(fold_path, "out"), exist_ok=True)

        model = load_model(os.path.join(fold_path, "model.keras"), compile=False)
        

        with open(os.path.join(fold_path, "train_args.json"), 'r') as json_file:
            train_dict = json.load(json_file)
            
        task = train_dict["task"]
        model_name = train_dict["model_name"]
        fold_idx = train_dict["fold_idx"]
        pretrained = train_dict["pretrained"]
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
            heatmaps = []
            intersections_in = []
            intersections_out = []
            print(row["new_filename"])
            img = cv2.imread(os.path.join(database, row["new_filename"]), 0)
            img = np.expand_dims(img, axis=-1)
            
            mask = cv2.imread(os.path.join(MASKS_PATH, row["new_filename"]), 0)
            mask = cv2.resize(mask, (512, 512))
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            
            if database == CROPPED_DATABASE_PATH: 
                img = preprocess(img, mask=mask)
            else: 
                img = preprocess(img)   

            X = img.copy()
            X = np.expand_dims(X, axis=0)
            predicted_probs = model.predict(X)
            predicted_category_index = np.argmax(predicted_probs)
            print("PREDICTED_INDEX: ", predicted_category_index)
            score = CategoricalScore([predicted_category_index])

            # ------------------- GradCAM -------------------
            gradcam = Gradcam(model,
                            clone=True)

            cam = gradcam(score,
                        X,
                        penultimate_layer=LAST_CONV_LAYERS[model_name], 
                        seek_penultimate_conv_layer=False
                        )
            heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)   
            heatmaps.append(heatmap)
            
            _, th_heatmap = cv2.threshold(cam[0], 0.5, 1, cv2.THRESH_BINARY)
            th_heatmap = (th_heatmap * 255).astype(np.uint8)
            heatmap_in = cv2.bitwise_and(mask, th_heatmap)
            intersections_in.append(heatmap_in)
            heatmap_out = cv2.bitwise_and(cv2.bitwise_not(mask), th_heatmap)
            intersections_out.append(heatmap_out)


            # ------------------- GradCAM++ -------------------
            gradcam = GradcamPlusPlus(
                model,
                clone=True
            )

            cam = gradcam(score,
                        X,
                        penultimate_layer=LAST_CONV_LAYERS[model_name],
                        seek_penultimate_conv_layer=False,
                        )
            heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
            heatmaps.append(heatmap)

            _, th_heatmap = cv2.threshold(cam[0], 0.5, 1, cv2.THRESH_BINARY)
            th_heatmap = (th_heatmap * 255).astype(np.uint8)
            heatmap_in = cv2.bitwise_and(mask, th_heatmap)
            intersections_in.append(heatmap_in)
            heatmap_out = cv2.bitwise_and(cv2.bitwise_not(mask), th_heatmap)
            intersections_out.append(heatmap_out)

            # ------------------- ScoreCAM -------------------
            scorecam = Scorecam(
                model, 
                clone=True
            )

            # Generate heatmap with ScoreCAM
            cam = scorecam(
                score,
                X,
                penultimate_layer=LAST_CONV_LAYERS[model_name],
                seek_penultimate_conv_layer=False, 
                # max_N=10 # Faster-ScoreCAM
            )
            heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
            heatmaps.append(heatmap)
            
            _, th_heatmap = cv2.threshold(cam[0], 0.5, 1, cv2.THRESH_BINARY)
            th_heatmap = (th_heatmap * 255).astype(np.uint8)
            heatmap_in = cv2.bitwise_and(mask, th_heatmap)
            intersections_in.append(heatmap_in)
            heatmap_out = cv2.bitwise_and(cv2.bitwise_not(mask), th_heatmap)
            intersections_out.append(heatmap_out)
            
            # ------------------- LIME -------------------
            # lime_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            lime_img = img.copy()
            lime_img = np.squeeze(lime_img)
            print(lime_img.shape) 

            explanation = explainer.explain_instance(
                lime_img, 
                lime_predict, 
                random_seed=42, 
                top_labels=2, 
                hide_color=0, 
                num_samples=1000, 
                batch_size=32
            )
            temp, lime_mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False, min_weight=0.)
            
            lime_image = mark_boundaries(temp / 2 + 0.5, lime_mask)
            heatmaps.append(lime_image)
            lime_mask = (lime_mask * 255).astype(np.uint8)
            heatmap_in = cv2.bitwise_and(mask, lime_mask)
            intersections_in.append(heatmap_in)
            heatmap_out = cv2.bitwise_and(cv2.bitwise_not(mask), lime_mask)
            intersections_out.append(heatmap_out)

            # ------------------- Render -------------------
            methods = ["GradCam", "GradCam++", "ScoreCam", "Lime"]
            f, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 5))
            for i, (title, heatmap) in enumerate(zip(methods, heatmaps)):
                if title == "Lime":                            
                    ax[i].set_title("Lime", fontsize=16)
                    ax[i].imshow(heatmap, cmap="gray")
                    ax[i].axis('off')
                else: 
                    ax[i].set_title(title, fontsize=16)
                    ax[i].imshow(img, cmap="gray")
                    ax[i].imshow(heatmap, cmap='jet', alpha=0.3)
                    ax[i].axis('off')
                
            plt.tight_layout()
            f.savefig(os.path.join(figure_path, row["new_filename"]))
            
            f, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 5))
            for i, (title, intersection) in enumerate(zip(methods, intersections_in)):
                ax[i].set_title(title, fontsize=16)
                ax[i].imshow(intersection, cmap="gray")
                ax[i].axis('off')
                
            plt.tight_layout()
            f.savefig(os.path.join(os.path.join(fold_path, "in"), row["new_filename"]))

            f, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 5))
            for i, (title, intersection) in enumerate(zip(methods, intersections_out)):
                ax[i].set_title(title, fontsize=16)
                ax[i].imshow(intersection, cmap="gray")
                ax[i].axis('off')
                
            plt.tight_layout()
            f.savefig(os.path.join(os.path.join(fold_path, "out"), row["new_filename"]))