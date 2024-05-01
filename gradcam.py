import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import numpy as np
from keras.models import load_model
# from data_loader import preprocess
from utils import load_encoder
from models.utils import get_preprocessing_func
from gradcamplus import grad_cam_plus

import cv2
from paths import *
from natsort import natsorted
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import Model

LAST_CONV_LAYERS = {
    "xception": "block14_sepconv2_act",
    "mobilenet": "multiply_17", 
    "densenet": "relu"
}


def show_imgwithheat(img_path, heatmap, alpha=0.4, return_array=False):
    """Show the image with heatmap.

    Args:
        img_path: string.
        heatmap: image array, get it by calling grad_cam().
        alpha: float, transparency of heatmap.
        return_array: bool, return a superimposed image array or not.
    Return:
        None or image array.
    """
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (heatmap*255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    
    return superimposed_img, heatmap

def grad_cam(model, img,
             layer_name="block14_sepconv2_act",
             category_id=None):
    """Get a heatmap by Grad-CAM.

    Args:
        model: A model object, build from tf.keras 2.X.
        img: An image ndarray.
        layer_name: A string, layer name in model.
        category_id: An integer, index of the class.
            Default is the category with the highest score in the prediction.

    Return:
        A heatmap ndarray(without color).
    """
    img_tensor = np.expand_dims(img, axis=0)

    conv_layer = model.get_layer(layer_name)
    heatmap_model = Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(img_tensor)
        if category_id is None:
            category_id = np.argmax(predictions[0])
        output = predictions[:, category_id]
        # print("OUTPUT: ", output, output.shape)
        one_hot_pred = np.zeros_like(predictions[0])
        one_hot_pred[category_id] = 1
        # print("PRED: ", encoder.inverse_transform(one_hot_pred.reshape(1, -1)))
        
        grads = gtape.gradient(output, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    return np.squeeze(heatmap)

if __name__ == "__main__":
    # train_list = natsorted(os.listdir(TRAIN_PATH))
    train_list = ["0_model_name_xception_task_detection_pretrained_True_trainable_core_True_optimizer_sgd_epochs_300_lr_0.001_patience_5_batch_size_64_augmentation_prob_0.0_database_noncropped_top_idx_1_loss_categorical_crossentropy"]
    for train_folder in train_list: 
        print(f"Starting gradcam: {train_folder}")
        for fold_idx in range(1, 2):
            fold_path = os.path.join(TRAIN_PATH, train_folder, str(fold_idx))
            print(f"Fold idx: {fold_idx}")
            # if os.path.exists(os.path.join(fold_path, "gradcam")):
            #     continue
            os.makedirs(os.path.join(fold_path, "gradcam"), exist_ok=True)
            
            # if os.path.exists(os.path.join(fold_path, "gradcam_plus")):
            #     continue
            os.makedirs(os.path.join(fold_path, "gradcam_plus"), exist_ok=True)
            
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
                                 
                heatmap = grad_cam(model, img, layer_name=LAST_CONV_LAYERS[model_name])
                
                ol_image, heatmap = show_imgwithheat(os.path.join(database, row["new_filename"]), heatmap)
                ol_image = cv2.cvtColor(ol_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(fold_path, "gradcam", row["new_filename"]), ol_image)
                
                heatmap = grad_cam_plus(model, img, layer_name=LAST_CONV_LAYERS[model_name])
                ol_image, heatmap = show_imgwithheat(os.path.join(database, row["new_filename"]), heatmap)
                ol_image = cv2.cvtColor(ol_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(fold_path, "gradcam_plus", row["new_filename"]), ol_image)
                
                # print(ol_image.shape)
                # cv2.imwrite(os.path.join(IMAGES_CHECK, row["new_filename"]), heatmap)

            else: 
                print(f"Fold {fold_idx} done!")
                del model
                
        else: 
            print(f"Folder {train_folder} done!")