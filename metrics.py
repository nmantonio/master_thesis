from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

def extract_true_label(filename):
    if filename.split('_')[1] != "normal":
        return 0
    else: 
        return 1

# pred_path = r"/home/marc/ANTONIO_EXPERIMENTS/TFM/master_thesis/predictions.csv"
pred_path = r"/home/antonio/Desktop/Antonio/master_thesis/predictions.csv"

pred_df = pd.read_csv(pred_path)

pred_df["binary_prediction"] = pred_df["prediction"].apply(round)

pred_df["true_label"] = pred_df["filename"].apply(extract_true_label)

metrics = dict(
    accuracy=accuracy_score(y_true=pred_df["true_label"], y_pred=pred_df["binary_prediction"]), 
    precision=precision_score(y_true=pred_df["true_label"], y_pred=pred_df["binary_prediction"]),
    recall=recall_score(y_true=pred_df["true_label"], y_pred=pred_df["binary_prediction"]),
    f1_score=f1_score(y_true=pred_df["true_label"], y_pred=pred_df["binary_prediction"])
)

for key, value in metrics.items():
    print(f"{key}: {round(value, 4)}")