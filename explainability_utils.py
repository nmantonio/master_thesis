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

from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam import Gradcam, GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam


LAST_CONV_LAYERS = {
    "xception": "block14_sepconv2_act",
    "mobilenet": "multiply_17", 
    "densenet": "relu"
}

train_list = ["2_model_name_densenet_task_detection_pretrained_True_trainable_core_True_optimizer_sgd_epochs_300_lr_0.001_patience_5_batch_size_32_augmentation_prob_0.0_database_noncropped_top_idx_1_loss_categorical_crossentropy"]
for train_folder in train_list: 
    print(f"Starting gradcam: {train_folder}")
    for fold_idx in range(1, 2):
        fold_path = os.path.join(TRAIN_PATH, train_folder, str(fold_idx))
        print(f"Fold idx: {fold_idx}")
        
        model = load_model(os.path.join(fold_path, "model.keras"), compile=False)
        

        with open(os.path.join(fold_path, "train_args.json"), 'r') as json_file:
            train_dict = json.load(json_file)
            
        task = train_dict["task"]
        model_name = train_dict["model_name"]
        print(model_name)
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
            print(row["new_filename"])
            img = cv2.imread(os.path.join(database, row["new_filename"]), 0)
            img = np.expand_dims(img, axis=-1)
            
            if database == CROPPED_DATABASE_PATH: 
                mask = cv2.imread(os.path.join(MASKS_PATH, row["new_filename"]), 0)
                mask = cv2.resize(mask, (512, 512))
                img = preprocess(img, mask=mask)
            else: 
                img = preprocess(img)   
                print(np.unique(img))


            X = img.copy()
            X = np.expand_dims(X, axis=0)
            predicted_probs = model.predict(X)
            predicted_category_index = np.argmax(predicted_probs)
            print("PREDICTED_INDEX: ", predicted_category_index)
            

            replace2linear = ReplaceToLinear()

            # Create Saliency object.
            saliency = Saliency(model,
                                model_modifier=replace2linear,
                                clone=True)

            # Generate saliency map
            score = CategoricalScore([predicted_category_index])

            saliency_map = saliency(score, X)
            heatmaps.append(saliency_map[0])

            # Generate saliency map with smoothing that reduce noise by adding noise
            saliency_map = saliency(score,
                                    X,
                                    smooth_samples=20, # The number of calculating gradients iterations.
                                    smooth_noise=0.20) # noise spread level.
            print(saliency_map.shape)
            print("Saliency", np.min(saliency_map), np.max(saliency_map), np.unique(saliency_map))
            heatmaps.append(saliency_map[0]*255)

            ## Since v0.6.0, calling `normalize()` is NOT necessary.
            # saliency_map = normalize(saliency_map)

            # Create Gradcam object
            gradcam = Gradcam(model,
                            model_modifier=replace2linear,
                            clone=True)

            # Generate heatmap with GradCAM
            cam = gradcam(score,
                        X,
                        penultimate_layer=LAST_CONV_LAYERS[model_name], 
                        seek_penultimate_conv_layer=False
                        )
            print(cam.shape)
            heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)  
            print("Gradcam", np.min(heatmap), np.max(heatmap), np.unique(heatmap))
 
            heatmaps.append(heatmap)


            ## Since v0.6.0, calling `normalize()` is NOT necessary.
            # cam = normalize(cam)


            # Create GradCAM++ object
            gradcam = GradcamPlusPlus(model,
                                    model_modifier=replace2linear,
                                    clone=True)

            # Generate heatmap with GradCAM++
            cam = gradcam(score,
                        X,
                        penultimate_layer=LAST_CONV_LAYERS[model_name],
                        seek_penultimate_conv_layer=False,
                        )
            print(cam.shape)

            heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
            print("GradcamPlusPlus", np.min(heatmap), np.max(heatmap), np.unique(heatmap))
            heatmaps.append(heatmap)


            ## Since v0.6.0, calling `normalize()` is NOT necessary.
            # cam = normalize(cam)



            # Create ScoreCAM object
            scorecam = Scorecam(model)

            # Generate heatmap with ScoreCAM
            cam = scorecam(score, X, penultimate_layer=LAST_CONV_LAYERS[model_name], seek_penultimate_conv_layer=False)
            print(cam.shape)

            heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
            print("ScoreCam", np.min(heatmap), np.max(heatmap), np.unique(heatmap))
            heatmaps.append(heatmap)
            ## Since v0.6.0, calling `normalize()` is NOT necessary.
            # cam = normalize(cam)


            # Create ScoreCAM object
            scorecam = Scorecam(model,
                                model_modifier=replace2linear
                                )

            # Generate heatmap with Faster-ScoreCAM
            cam = scorecam(score,
                        X,
                        penultimate_layer=LAST_CONV_LAYERS[model_name],
                        seek_penultimate_conv_layer=False,
                        max_N=10)  
            print(cam.shape)

            heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
            print("FScoreCam", np.min(heatmap), np.max(heatmap), np.unique(heatmap))
            heatmaps.append(heatmap)


            ## Since v0.6.0, calling `normalize()` is NOT necessary.
            # cam = normalize(cam)

            # Render
            methods = ["Saliency", "Smooth", "GradCam", "GradCam++", "ScoreCam", "NScoreCam"]
            f, ax = plt.subplots(nrows=1, ncols=6, figsize=(24, 4))
            for i, (title, heatmap) in enumerate(zip(methods, heatmaps)):
                print(heatmap.shape)

                ax[i].set_title(title, fontsize=16)
                ax[i].imshow(img, cmap="gray")
                ax[i].imshow(heatmap, cmap='jet', alpha=0.3)
                ax[i].axis('off')
            plt.tight_layout()
            f.savefig(fr"/home/anadal/Experiments/TFM/trains/0_model_name_xception_task_detection_pretrained_True_trainable_core_True_optimizer_sgd_epochs_300_lr_0.001_patience_5_batch_size_64_augmentation_prob_0.0_database_noncropped_top_idx_1_loss_categorical_crossentropy/1/xai/{row['new_filename']}.png")
