import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import json
import random

from natsort import natsorted
from paths import *
import cv2
import pandas as pd
from utils import load_encoder
from models.utils import get_preprocessing_func
from keras.models import load_model

from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.gradcam import Gradcam, GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam

LAST_CONV_LAYERS = {
    "xception": "block14_sepconv2_act",
    "mobilenet": "multiply_17", 
    "densenet": "relu"
}
train_list = natsorted([name for name in os.listdir(TRAIN_PATH) if "noncropped" in name]) # train_0
# train_list = natsorted([name for name in os.listdir(TRAIN_PATH) if "noncropped" in name])[::-1] # train_1

for train_folder in train_list: 
    print(f"Starting coverage: {train_folder}")
    coverage_resume = {method: [] for method in ["gradcam", "gradcam++", "scorecam"]}
    for fold_idx in range(1, 6):
        random.seed(42)
        
        fold_coverage = {}
        fold_path = os.path.join(TRAIN_PATH, train_folder, str(fold_idx))
        print(f"Fold idx: {fold_idx}")
        
        figure_path = os.path.join(fold_path, "explainability_figures")
        # if os.path.exists(figure_path):
        #     continue
        os.makedirs(figure_path, exist_ok=True)
        # os.makedirs(os.path.join(fold_path, "in"), exist_ok=True)
        # os.makedirs(os.path.join(fold_path, "out"), exist_ok=True)
        print("Loading model...")
        model = load_model(os.path.join(fold_path, "model.keras"), compile=False)
        print("Model loaded!")

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
        elif task == "abnormal_classification":
            encoder = load_encoder(ABNORMAL_CLASSIFICATION_ENCODER)
            df = pd.read_csv(CLASSIFICATION_CSV)
            df = df[df["classification"] != "normal"]
            task = "classification"

        df = df[df["split"] == str(fold_idx)]
        
        print("Starting image explanation...")
        for idx, row in df.iterrows():
            
            fold_coverage[row["new_filename"]] = {}

            
            heatmaps = []
            intersections_in = []

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
                 
            total_area = np.sum(th_heatmap)
            if total_area == 0:
                coverage = 0
            else:
                coverage = np.sum(heatmap_in) / (total_area) * 100
            fold_coverage[row["new_filename"]]["gradcam"] = coverage
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

            total_area = np.sum(th_heatmap)
            if total_area == 0:
                coverage = 0
            else:
                coverage = np.sum(heatmap_in) / (total_area) * 100
            fold_coverage[row["new_filename"]]["gradcam++"] = coverage
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
                max_N=10 # Faster-ScoreCAM
            )
            heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
            heatmaps.append(heatmap)
            
            _, th_heatmap = cv2.threshold(cam[0], 0.5, 1, cv2.THRESH_BINARY)
            th_heatmap = (th_heatmap * 255).astype(np.uint8)
            heatmap_in = cv2.bitwise_and(mask, th_heatmap)
            intersections_in.append(heatmap_in)
            
            total_area = np.sum(th_heatmap)
            if total_area == 0:
                coverage = 0
            else:
                coverage = np.sum(heatmap_in) / (total_area) * 100
            fold_coverage[row["new_filename"]]["scorecam"] = coverage
            #             # ------------------- Faster ScoreCAM -------------------
            # scorecam = Scorecam(
            #     model, 
            #     clone=True
            # )

            # # Generate heatmap with ScoreCAM
            # cam = scorecam(
            #     score,
            #     X,
            #     penultimate_layer=LAST_CONV_LAYERS[model_name],
            #     seek_penultimate_conv_layer=False, 
            #     max_N=10 # Faster-ScoreCAM
            # )
            # heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)
            # heatmaps.append(heatmap)
            
            # _, th_heatmap = cv2.threshold(cam[0], 0.5, 1, cv2.THRESH_BINARY)
            # th_heatmap = (th_heatmap * 255).astype(np.uint8)
            # heatmap_in = cv2.bitwise_and(mask, th_heatmap)
            # intersections_in.append(heatmap_in)
            # heatmap_out = cv2.bitwise_and(cv2.bitwise_not(mask), th_heatmap)
            # intersections_out.append(heatmap_out)
            # fasterscorecam_time = time()
                
            # ------------------- Render -------------------
            # methods = ["GradCam", "GradCam++", "ScoreCam"]
            # f, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
            # for i, (title, heatmap) in enumerate(zip(methods, heatmaps)):
            #     ax[i].set_title(title, fontsize=16)
            #     ax[i].imshow(img, cmap="gray")
            #     ax[i].imshow(heatmap, cmap='jet', alpha=0.3)
            #     ax[i].axis('off')
                
            # plt.tight_layout()
            # f.savefig(os.path.join(figure_path, row["new_filename"]))
            

            # Perform the action with a 10% probability
            random_number = random.randint(0, 99)
            if random_number < 10:
                methods = ["GradCam", "GradCam++", "ScoreCam"]
                img = cv2.imread(os.path.join(database, row["new_filename"]), 0)

                for method_name, heatmap in zip(methods, heatmaps):
                    combined_image = 0.7 * cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) + 0.3*cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
                    output_filename = os.path.join(figure_path, f"{method_name}_{row['new_filename']}")
                    cv2.imwrite(output_filename, combined_image)
                    
                    
            # f, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 5))
            # for i, (title, intersection) in enumerate(zip(methods, intersections_in)):
            #     ax[i].set_title(title, fontsize=16)
            #     ax[i].imshow(intersection, cmap="gray")
            #     ax[i].axis('off')
                
            # plt.tight_layout()
            # f.savefig(os.path.join(os.path.join(fold_path, "in"), row["new_filename"]))

            # f, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 5))
            # for i, (title, intersection) in enumerate(zip(methods, intersections_out)):
            #     ax[i].set_title(title, fontsize=16)
            #     ax[i].imshow(intersection, cmap="gray")
            #     ax[i].axis('off')
                
            # plt.tight_layout()
            # f.savefig(os.path.join(os.path.join(fold_path, "out"), row["new_filename"]))
            
        for method_name in ['gradcam', "gradcam++", "scorecam"]:
            method_values = np.array([fold_coverage[key][method_name] for key in fold_coverage.keys()])
            mean_coverage = np.mean(method_values)
            coverage_resume[method_name].append(mean_coverage)
        
        with open(os.path.join(fold_path, "val", "coverage_by_filename.json"), "w") as json_file:
            json.dump(fold_coverage, json_file, indent=4)
        
    # Save coverage_resume to a JSON file
    with open(os.path.join(fold_path, "coverage_resume.json"), "w") as json_file:
        json.dump(coverage_resume, json_file, indent=4)
    
    # Save overall coverage to a JSON file
    overall_coverage = {method: np.mean(coverages) for method, coverages in coverage_resume.items()}
    with open(os.path.join(TRAIN_PATH, train_folder, "overall_coverage.json"), "w") as json_file:
        json.dump(overall_coverage, json_file, indent=4)