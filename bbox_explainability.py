import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pydicom
import cv2
import json
import numpy as np
import pandas as pd
from matplotlib import cm
from keras.models import load_model

from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.gradcam import Gradcam, GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam

# from lime import lime_image
# explainer = lime_image.LimeImageExplainer()
# from skimage.segmentation import mark_boundaries



# def lime_predict(image, save_input_example=False):
#     image = image.astype(np.float32)
#     if image.shape[-1] == 3:
#         image = np.array([cv2.cvtColor(image[idx], cv2.COLOR_BGR2GRAY) for idx in range(image.shape[0])])
#     if len(image.shape) == 3:
#         image = np.expand_dims(image, axis=-1)
    
#     # if save_input_example:
#     #     for idx in range(0, image.shape[0]):
#     #         cv2.imwrite(os.path.join(save_input_example, f"image{idx}.png"),image[idx]*255)
#     return c_model.predict(image)

from models.utils import get_preprocessing_func
from utils import load_encoder
from paths import GIT_PATH, TRAIN_PATH, DATABASE_PATH, CROPPED_DATABASE_PATH, MASKS_PATH, CLASSIFICATION_CSV, BBOX_CSV
from paths import DETECTION_ENCODER, ABNORMAL_CLASSIFICATION_ENCODER

ad_folder = r"29_model_name_densenet_task_detection_pretrained_True_trainable_core_True_optimizer_sgd_epochs_300_lr_0.001_patience_5_batch_size_32_augmentation_prob_0.25_database_cropped_top_idx_1_loss_categorical_crossentropy" # Abnormality detection model chosen
c_folder = r"77_model_name_densenet_task_abnormal_classification_pretrained_True_trainable_core_True_optimizer_sgd_epochs_300_lr_0.001_patience_5_batch_size_32_augmentation_prob_0.25_database_cropped_top_idx_1_loss_categorical_crossentropy" # Multiclass 6class classification model chosen<


for fold_idx in range(1, 6):
    save_path = os.path.join(GIT_PATH, "two_stage_test", str(fold_idx), "bbox")
    figure_path = os.path.join(save_path, "figures")
    c_image_path = os.path.join(save_path, "c_image")
    os.makedirs(figure_path, exist_ok=True)
    os.makedirs(c_image_path, exist_ok=True)

    ad_fold_path = os.path.join(TRAIN_PATH, ad_folder, str(fold_idx))
    c_fold_path = os.path.join(TRAIN_PATH, c_folder, str(fold_idx))
    
    # Get only test data and create results df
    df = pd.read_csv(CLASSIFICATION_CSV)
    df = df[df["split"] == "test"]
    df = df[df["classification"] == "pulmonary_fibrosis"]
    df = df[df["database"] == "DS6"]
    
    df_bbox = pd.read_csv(BBOX_CSV)
    df_bbox = df_bbox[df_bbox["class_id"] == 13]

    
    # Abnormality detection loading
    print("Abnormality loading...")
    with open(os.path.join(ad_fold_path, "train_args.json"), 'r') as json_file:
        data = json.load(json_file)
    
    model_name = data["model_name"]
    pretrained = data["pretrained"]
    task = data["task"]
    assert task == "detection", f"Task is {task}, not detection!"
    
    database = data["database"]
    assert database == "cropped", f"Database training param is {database}, not cropped!"
    database = CROPPED_DATABASE_PATH
    
    ad_preprocessing = get_preprocessing_func(name=model_name, pretrained=pretrained, task=task, database=database)
    ad_encoder = load_encoder(DETECTION_ENCODER)
    ad_model = load_model(os.path.join(ad_fold_path, "model.keras"))
    print("Abnormality model: OK!")
    
    # Disease classification loading
    print("Disease classification loading...")
    with open(os.path.join(c_fold_path, "train_args.json"), "r") as json_file:
        data = json.load(json_file)
        
    model_name = data["model_name"]
    pretrained = data["pretrained"]
    task = data["task"]
    assert task == "abnormal_classification", f"Task is {task}, not abnormal_classification!"
    
    database = data["database"]
    assert database == "cropped", f"Database training param is {database}, not cropped!"
    database = CROPPED_DATABASE_PATH
    
    c_preprocessing = get_preprocessing_func(name=model_name, pretrained=pretrained, task=task, database=database)
    c_encoder = load_encoder(ABNORMAL_CLASSIFICATION_ENCODER)
    c_model = load_model(os.path.join(c_fold_path, "model.keras"))
    print("Abnormality model: OK!")
    
    # Loading explainability models
    gradcam_model = Gradcam(c_model, clone=True)
    gradcamplusplus_model = GradcamPlusPlus(c_model, clone=True)
    scorecam_model = Scorecam(c_model, clone=True)
        
    # Starting predictions
    print("Starting predictions...")
    test_df = pd.DataFrame(columns = ["new_filename", "true", "pred"])

    coverage_values = {key: [] for key in ('gradcam', 'gradcamplusplus', 'scorecam')}
    for idx, row in df.iterrows():
        print(f"True label: {row['classification']}")
        image = cv2.imread(os.path.join(database, row["new_filename"]), 0)
        image = np.expand_dims(image, axis=-1)
        
        mask = cv2.imread(os.path.join(MASKS_PATH, row["new_filename"]), 0)
        mask = cv2.resize(mask, (512, 512))
        ad_image = ad_preprocessing(image, mask=mask)

        ad_image = np.expand_dims(ad_image, axis=0)

        image_pred = ad_model.predict(ad_image, verbose=0)[0]
        
        one_hot_pred = np.zeros_like(image_pred)
        one_hot_pred[np.argmax(image_pred)] = 1
        predicted_label = ad_encoder.inverse_transform(one_hot_pred.reshape(1, -1))
    
        if predicted_label[0][0] == "abnormal":
            print("Abnormality detected... classifing abnormality!")
            c_image = c_preprocessing(image, mask=mask)
            c_image = np.expand_dims(c_image, axis=0)
            image_pred = c_model.predict(c_image, verbose=0)[0]
            
            one_hot_pred = np.zeros_like(image_pred)
            one_hot_pred[np.argmax(image_pred)] = 1
            predicted_label = c_encoder.inverse_transform(one_hot_pred.reshape(1, -1))
            print(predicted_label)
        
        print(f"Predicted: {predicted_label[0][0]}")
        
        if predicted_label[0][0] == "pulmonary_fibrosis":
            # save_c_image = c_image[0].copy()
            # save_c_image = (save_c_image - np.min(save_c_image)) / (np.max(save_c_image) - np.min(save_c_image)) * 255
            cv2.imwrite(os.path.join(c_image_path, row["new_filename"]), c_image[0]*255)
            image = cv2.imread(os.path.join(DATABASE_PATH, row["new_filename"]), 0)

            # Get binary rectangle mask
            name = ".".join(row["original_filename"].split('.')[:-1])
            aux = df_bbox[df_bbox["image_id"] == name]
            dcm_name = ".".join(row["original_filename"].split('.')[:-1]) + ".dicom"

            dcm_image = pydicom.dcmread(os.path.join(r"/home/anadal/Experiments/TFM/dcm_temp", dcm_name))
            shape = dcm_image.pixel_array.shape
            print(shape)
            
            rect_mask = np.zeros((512, 512), dtype=np.uint8)
            
            # shape = (df.loc[idx, "shape2"], df.loc[idx, "shape1"])
            # print(shape)
            for idx_2, row_2 in aux.iterrows():
                x_min, y_min = int((row_2["x_min"]*512) // shape[1]), int((row_2["y_min"]*512) // shape[0])
                x_max, y_max = int((row_2["x_max"]*512) // shape[1]), int((row_2["y_max"]*512) // shape[0])
                print(x_min, y_min, x_max, y_max)
                
                color = (0, 0, 255)  # BGR color format, here it's green
                thickness = 2  # Thickness of the rectangle border      
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

            cv2.rectangle(rect_mask, (x_min, y_min), (x_max, y_max), 255, -1)

            # Get explainability explanations
            score = CategoricalScore([np.argmax(image_pred)])
            
            cam_gradcam = gradcam_model(
                score, 
                c_image, 
                penultimate_layer = "relu", 
                seek_penultimate_conv_layer=False
            )

            heatmap_gradcam = np.uint8(cm.jet(cam_gradcam[0])[..., :3]*255)
            _, th_heatmap_gradcam = cv2.threshold(cam_gradcam[0], 0.5, 1, cv2.THRESH_BINARY)
            th_heatmap_gradcam = (th_heatmap_gradcam * 255).astype(np.uint8)

            heatmap_in = cv2.bitwise_and(rect_mask, th_heatmap_gradcam)
            total_area = np.sum(th_heatmap_gradcam)            
            if total_area == 0:
                coverage = 0
            else:
                coverage = np.sum(heatmap_in) / (total_area) * 100
            coverage_values["gradcam"].append(coverage)
            
            img = image.copy()
            combined_image = 0.7 * cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) + 0.3*cv2.cvtColor(heatmap_gradcam, cv2.COLOR_RGB2BGR)
            output_filename = os.path.join(figure_path, f"GradCam_{row['new_filename']}")
            cv2.imwrite(output_filename, combined_image)
            
            # Gradcamplusplus
            cam_gradcamplusplus = gradcamplusplus_model(
                score, 
                c_image, 
                penultimate_layer = "relu", 
                seek_penultimate_conv_layer=False
            )

            heatmap_gradcamplusplus = np.uint8(cm.jet(cam_gradcamplusplus[0])[..., :3]*255)
            _, th_heatmap_gradcamplusplus = cv2.threshold(cam_gradcamplusplus[0], 0.5, 1, cv2.THRESH_BINARY)
            th_heatmap_gradcamplusplus = (th_heatmap_gradcamplusplus * 255).astype(np.uint8)

            heatmap_in = cv2.bitwise_and(rect_mask, th_heatmap_gradcamplusplus)
            total_area = np.sum(th_heatmap_gradcamplusplus)            
            if total_area == 0:
                coverage = 0
            else:
                coverage = np.sum(heatmap_in) / (total_area) * 100
            print(coverage)
            coverage_values["gradcamplusplus"].append(coverage)

            
            img = image.copy()
            combined_image = 0.7 * cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) + 0.3*cv2.cvtColor(heatmap_gradcamplusplus, cv2.COLOR_RGB2BGR)
            output_filename = os.path.join(figure_path, f"Gradcam++{row['new_filename']}")
            cv2.imwrite(output_filename, combined_image)
            
            # Scorecam
            cam_scorecam = scorecam_model(
                score, 
                c_image, 
                penultimate_layer = "relu", 
                seek_penultimate_conv_layer=False, 
                max_N=10
            )

            heatmap_scorecam = np.uint8(cm.jet(cam_scorecam[0])[..., :3]*255)
            _, th_heatmap_scorecam = cv2.threshold(cam_scorecam[0], 0.5, 1, cv2.THRESH_BINARY)
            th_heatmap_scorecam = (th_heatmap_scorecam * 255).astype(np.uint8)

            heatmap_in = cv2.bitwise_and(rect_mask, th_heatmap_scorecam)
            total_area = np.sum(th_heatmap_scorecam)            
            if total_area == 0:
                coverage = 0
            else:
                coverage = np.sum(heatmap_in) / (total_area) * 100
            print(coverage)
            coverage_values["scorecam"].append(coverage)

            
            img = image.copy()
            combined_image = 0.7 * cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) + 0.3*cv2.cvtColor(heatmap_scorecam, cv2.COLOR_RGB2BGR)
            output_filename = os.path.join(figure_path, f"ScoreCam{row['new_filename']}")
            cv2.imwrite(output_filename, combined_image)
            
    with open(os.path.join(save_path, "bbox_coverage.json"), "w") as json_file:
        json.dump(coverage_values, json_file, indent=4)
    
    mean_coverage = {key: np.mean(value) for key, value in coverage_values.items()}        
    with open(os.path.join(save_path, "mean_bbox_coverage.json"), "w") as json_file:
        json.dump(mean_coverage, json_file, indent=4)
            
            # lime_img = c_image.copy()
            # lime_img = np.squeeze(lime_img)
            # print(lime_img.shape) 

            # explanation = explainer.explain_instance(
            #     lime_img, 
            #     lime_predict, 
            #     random_seed=42, 
            #     top_labels=2, 
            #     hide_color=0, 
            #     num_samples=500, 
            #     batch_size=32
            # )
            # temp, lime_mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False, min_weight=0.)
            
            # img = image.copy()
            # lime_image = mark_boundaries((img/255) / 2 + 0.5, lime_mask)

            # lime_mask = (lime_mask * 255).astype(np.uint8)
            
            # heatmap_in = cv2.bitwise_and(rect_mask, lime_mask)
            # total_area = np.sum(lime_mask)            
            # if total_area == 0:
            #     coverage = 0
            # else:
            #     coverage = np.sum(heatmap_in) / (total_area) * 100
            # print(coverage)

            # output_filename = os.path.join(figure_path, f"Lime{row['new_filename']}")
            # cv2.imwrite(output_filename, lime_image*255)
    
            


