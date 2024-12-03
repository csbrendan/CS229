
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

#/Users/brendanmurphy/Desktop/CS229/PROJECT/FINAL_PROJECT_CODE/results/results_base_zoom_thresh_pt1_300frames_20241203_195142.json
#/Users/brendanmurphy/Desktop/CS229/PROJECT/FINAL_PROJECT_CODE/results/results_stage2_zoom_thresh_pt1_300frames_20241203_194836.json

#/Users/brendanmurphy/Desktop/CS229/PROJECT/FINAL_PROJECT_CODE/results/results_base_cdf_thresh_pt1_300frames20241203_230544.json
#/Users/brendanmurphy/Desktop/CS229/PROJECT/FINAL_PROJECT_CODE/results/results_stage2_cdf_thresh_pt1_300frames20241203_231639.json

# Load the JSON data
with open("/Users/brendanmurphy/Desktop/CS229/PROJECT/FINAL_PROJECT_CODE/results/results_stage2_cdf_thresh_pt1_300frames20241203_231639.json", "r") as f:
    data = json.load(f)

# Extract relevant data
y_true = [entry["ground_truth"] for entry in data["video_results"]]
y_pred = [entry["predicted_fake"] for entry in data["video_results"]]
y_scores = [entry["mean_prediction"] for entry in data["video_results"]]

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Calculate F1 score
f1 = f1_score(y_true, y_pred)
print(f"F1 score: {f1:.4f}")

# Calculate ROC curve and AUC
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.4f}")

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


