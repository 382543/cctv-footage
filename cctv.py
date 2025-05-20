import os
import glob
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Path to output reports
OUTPUT_FOLDER = r"D:\DV Lab\FaceRecognitionProject\output"
report_files = sorted(glob.glob(os.path.join(OUTPUT_FOLDER, "cam*_report.csv")))

def print_metrics(y_true, y_pred, label):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"--- {label} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}\n")

all_true = []
all_pred = []

for report_path in report_files:
    df = pd.read_csv(report_path)
    cam_name = os.path.basename(report_path).replace("_report.csv", "")

    # Ground truth: whether a face was present
    y_true = df["Face Detected"].astype(int).tolist()

    # Prediction: whether the model matched the face to the target
    y_pred = df["Match Found"].astype(int).tolist()

    print_metrics(y_true, y_pred, cam_name)

    all_true.extend(y_true)
    all_pred.extend(y_pred)

# Overall summary
print("=== OVERALL METRICS ===")
print_metrics(all_true, all_pred, "All Cameras Combined")