import os
from paths import MONTGOMERY_MASKS, RAW_MASKS_PATH
import cv2
from sklearn.metrics import f1_score, jaccard_score
import numpy as np

# Get the list of mask filenames in the Montgomery dataset
montgomery_names = os.listdir(MONTGOMERY_MASKS)

# Lists to store the computed metrics
f1_scores_0 = []
jaccard_indices_0 = []
f1_scores_1 = []
jaccard_indices_1 = []
f1_scores_all = []
jaccard_indices_all = []

# Iterate over each mask in the Montgomery dataset
for name in montgomery_names:
    # Construct the modified name based on the condition
    if name.split("_")[-1].startswith("0"):
        mod_name = f"normal_DS2_{name}"
        is_zero = True
    else:
        mod_name = f"tuberculosis_DS2_{name}"
        is_zero = False

    # Read the ground truth mask and resize to (256, 256)
    gt = cv2.imread(os.path.join(MONTGOMERY_MASKS, name), 0)
    gt = cv2.resize(gt, (256, 256))

    # Read the predicted mask
    pred = cv2.imread(os.path.join(RAW_MASKS_PATH, mod_name), 0)

    # Ensure both masks are binary
    gt = (gt > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)

    # Compute F1-Score
    f1 = f1_score(gt.flatten(), pred.flatten())

    # Compute Jaccard Index
    jaccard = jaccard_score(gt.flatten(), pred.flatten())

    # Append metrics to respective lists based on the condition
    if is_zero:
        f1_scores_0.append(f1)
        jaccard_indices_0.append(jaccard)
    else:
        f1_scores_1.append(f1)
        jaccard_indices_1.append(jaccard)

    # Append metrics to overall lists
    f1_scores_all.append(f1)
    jaccard_indices_all.append(jaccard)

    # print(f"Image: {name}")
    # print(f"F1-Score: {f1}")
    # print(f"Jaccard Index: {jaccard}")
    # print()

# Display average metrics for images starting with '0'
average_f1_0 = np.mean(f1_scores_0)
average_jaccard_0 = np.mean(jaccard_indices_0)
print(f"Average F1-Score for images starting with '0': {average_f1_0}")
print(f"Average Jaccard Index for images starting with '0': {average_jaccard_0}")

# Display average metrics for images starting with '1'
average_f1_1 = np.mean(f1_scores_1)
average_jaccard_1 = np.mean(jaccard_indices_1)
print(f"Average F1-Score for images starting with '1': {average_f1_1}")
print(f"Average Jaccard Index for images starting with '1': {average_jaccard_1}")

# Display overall average metrics
average_f1_all = np.mean(f1_scores_all)
average_jaccard_all = np.mean(jaccard_indices_all)
print(f"Overall Average F1-Score: {average_f1_all}")
print(f"Overall Average Jaccard Index: {average_jaccard_all}")
