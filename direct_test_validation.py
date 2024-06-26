import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import json
import numpy as np
import pandas as pd
from keras import backend as K 
from keras.models import load_model

from models.utils import get_preprocessing_func
from utils import load_encoder
from paths import GIT_PATH, TRAIN_PATH, CROPPED_DATABASE_PATH, MASKS_PATH, CLASSIFICATION_CSV
from paths import CLASSIFICATION_ENCODER

train_folder = r"53_model_name_densenet_task_classification_pretrained_True_trainable_core_True_optimizer_sgd_epochs_300_lr_0.001_patience_5_batch_size_32_augmentation_prob_0.25_database_cropped_top_idx_1_loss_categorical_crossentropy"
    
for fold_idx in range(1, 6):
    save_path = os.path.join(GIT_PATH, "direct_test", str(fold_idx))
    os.makedirs(save_path, exist_ok=True)
    fold_path = os.path.join(TRAIN_PATH, train_folder, str(fold_idx))
    
    # Get only test data and create results df
    df = pd.read_csv(CLASSIFICATION_CSV)
    df = df[df["split"] == "test"]
    
    print("Direct classification model loading...")
    with open(os.path.join(fold_path, "train_args.json"), 'r') as json_file:
        data = json.load(json_file)
    
    model_name = data["model_name"]
    pretrained = data["pretrained"]
    task = data["task"]
    assert task == "classification", f"Task is {task}, not classification!"
    
    database = data["database"]
    assert database == "cropped", f"Database training param is {database}, not cropped!"
    database = CROPPED_DATABASE_PATH
    
    preprocessing = get_preprocessing_func(name=model_name, pretrained=pretrained, task=task, database=database)
    encoder = load_encoder(CLASSIFICATION_ENCODER)
    model = load_model(os.path.join(fold_path, "model.keras"))
    print("Direct classification model: OK!")
    
    
    # Starting predictions
    print("Starting predictions...")
    test_df = pd.DataFrame(columns = ["new_filename", "true", "pred"])

    predictions = {}
    for idx, row in df.iterrows():
        print(f"True label: {row[task]}")
        image = cv2.imread(os.path.join(database, row["new_filename"]), 0)
        image = np.expand_dims(image, axis=-1)
        
        mask = cv2.imread(os.path.join(MASKS_PATH, row["new_filename"]), 0)
        mask = cv2.resize(mask, (512, 512))
        image = preprocessing(image, mask=mask)

        image = np.expand_dims(image, axis=0)

        image_pred = model.predict(image, verbose=0)[0]
        
        one_hot_pred = np.zeros_like(image_pred)
        one_hot_pred[np.argmax(image_pred)] = 1
        predicted_label = encoder.inverse_transform(one_hot_pred.reshape(1, -1))

        print(f"Predicted: {predicted_label[0][0]}")
        predictions[row["new_filename"]]  = {
            "raw": image_pred.tolist(), 
            "one_hot": one_hot_pred.tolist(), 
            "pred_label": predicted_label[0][0], 
            "true_label": row['classification']
        }
        
        test_df = pd.concat([test_df, pd.DataFrame({"new_filename": [row["new_filename"]], "true": [row['classification']], "pred": [predicted_label[0][0]]})])

    # Store results
    test_df.to_csv(os.path.join(save_path, "test_results.csv"), index=False)
    with open(os.path.join(save_path, "test_raw.json"), "w") as json_file: 
        json.dump(predictions, json_file, indent=4)
            
            
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, fbeta_score
    cm = confusion_matrix(y_true=test_df["true"], y_pred=test_df["pred"], labels=test_df["true"].unique())
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_df["true"].unique())
    disp.plot(ax = ax, xticks_rotation=30, cmap="Blues")
    disp.figure_.savefig(os.path.join(save_path, "confusion_matrix.png"))

    # Pycm 
    from pycm import ConfusionMatrix
    cm = ConfusionMatrix(test_df["true"].tolist(), test_df["pred"].tolist())
    cm.save_obj(os.path.join(save_path, "cm_obj"), save_stat = True)
    cm.save_html(os.path.join(save_path, "report"))

    class_metrics = {
        "ACC": cm.class_stat["ACC"], 
        "TPR": cm.class_stat["TPR"], # recall, sensitivity
        "TNR": cm.class_stat["TNR"], # specificity
        "PPV": cm.class_stat["PPV"], # precision
        "F0.5": cm.class_stat["F0.5"],
        "F1": cm.class_stat["F1"], 
        "F2": cm.class_stat["F2"], 
        "GM": cm.class_stat["GM"], 
        "MCC": cm.class_stat["MCC"] # Matthews correlation coefficient
    }

    with open(os.path.join(save_path, "class_metrics.json"), "w") as json_file: 
        json.dump(class_metrics, json_file, indent=4)

    fold_metrics = {
        "Weighted Accuracy": cm.weighted_average("ACC"), 
        "Kappa": cm.overall_stat["Kappa"],
        "MCC": cm.overall_stat["Overall MCC"], 
        "TPR Micro": cm.overall_stat["TPR Micro"], 
        "TPR Macro": cm.overall_stat["TPR Macro"],
        "TNR Micro": cm.overall_stat["TNR Micro"], 
        "TNR Macro": cm.overall_stat["TNR Macro"],
        "PPV Micro": cm.overall_stat["PPV Micro"], 
        "PPV Macro": cm.overall_stat["PPV Macro"],
        "F1 Micro": cm.overall_stat["F1 Micro"], 
        "F1 Macro": cm.overall_stat["F1 Macro"], 
        "F0.5 Micro": fbeta_score(y_true=test_df["true"], y_pred=test_df["pred"], beta=0.5, average="micro"), 
        "F0.5 Macro": fbeta_score(y_true=test_df["true"], y_pred=test_df["pred"], beta=0.5, average="macro"), 
        "F2 Micro": fbeta_score(y_true=test_df["true"], y_pred=test_df["pred"], beta=2, average="micro"), 
        "F2 Macro": fbeta_score(y_true=test_df["true"], y_pred=test_df["pred"], beta=2, average="macro"),
    }
    if "normal" in cm.FP.keys():
        fold_metrics["Normal False Positives"] = cm.FP["normal"]


    with open(os.path.join(save_path, "fold_metrics.json"), "w") as json_file: 
        json.dump(fold_metrics, json_file, indent=4)
            
    K.clear_session()

