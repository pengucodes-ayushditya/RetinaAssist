import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.grading_models import APTOSGradeClassifier
from src.pipeline import RetinaDataset


CSV = Path("data/external/idrid/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv")
IMAGE_ROOT = Path("data/external/idrid/1. Original Images/b. Testing Set")
CHECKPOINT = Path("models/aptos_idrid_grade5_best.pt")
OUTPUT = Path("outputs/idrid_grade5_mixed_external.json")
GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def main():
    if not CSV.is_file():
        raise FileNotFoundError(f"IDRiD grading labels not found: {CSV}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"APTOS grade checkpoint not found: {CHECKPOINT}")

    labels = pd.read_csv(CSV)
    labels.columns = labels.columns.str.strip()
    name_col = "Image name"
    grade_col = "Retinopathy grade"
    if name_col not in labels or grade_col not in labels:
        raise ValueError(f"Expected columns {name_col!r} and {grade_col!r}; got {list(labels.columns)}")

    image_by_stem = {p.stem: p for p in IMAGE_ROOT.rglob("*") if p.is_file()}
    rows = []
    missing = []
    for _, row in labels.iterrows():
        image_name = str(row[name_col]).strip()
        image_path = image_by_stem.get(image_name)
        if image_path is None:
            missing.append(image_name)
            continue
        rows.append({"path": str(image_path), "label": int(row[grade_col])})

    if missing:
        raise FileNotFoundError(f"Could not find {len(missing)} labeled images; examples: {missing[:5]}")
    if not rows:
        raise ValueError("No labeled IDRiD testing images were found")

    frame = pd.DataFrame(rows)
    dataset = RetinaDataset(frame, train=False, size=224)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = APTOSGradeClassifier(pretrained=False).to(device)

    # This checkpoint was produced locally by this project and contains
    # validation metadata in addition to the model weights.
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels, all_probabilities = [], []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            all_probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            all_labels.extend(targets.numpy().tolist())

    y_true = np.asarray(all_labels, dtype=int)
    probabilities = np.concatenate(all_probabilities)
    y_pred = probabilities.argmax(axis=1)

    any_dr_score = 1.0 - probabilities[:, 0]
    referable_score = probabilities[:, 2:].sum(axis=1)
    any_dr_true = (y_true >= 1).astype(int)
    referable_true = (y_true >= 2).astype(int)
    threshold_config = json.loads(Path("outputs/aptos_idrid_grade5_thresholds.json").read_text())
    any_dr_threshold = threshold_config["any_dr"]["threshold"]
    referable_threshold = threshold_config["referable_dr_grade_ge_2"]["threshold"]
    any_dr_pred = (any_dr_score >= any_dr_threshold).astype(int)
    referable_pred = (referable_score >= referable_threshold).astype(int)

    report = {
        "dataset": "IDRiD official disease-grading testing split",
        "n_images": int(len(y_true)),
        "checkpoint": str(CHECKPOINT),
        "thresholds": {"any_dr": any_dr_threshold, "referable_dr_grade_ge_2": referable_threshold},
        "five_grade": {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "quadratic_weighted_kappa": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]).tolist(),
            "per_grade_f1": {
                name: float(f1_score(y_true == i, y_pred == i, zero_division=0))
                for i, name in enumerate(GRADE_NAMES)
            },
        },
        "any_dr": {
            "sensitivity": float(recall_score(any_dr_true, any_dr_pred, zero_division=0)),
            "specificity": float(recall_score(any_dr_true, any_dr_pred, pos_label=0, zero_division=0)),
            "confusion_matrix": confusion_matrix(any_dr_true, any_dr_pred, labels=[0, 1]).tolist(),
        },
        "referable_dr_grade_ge_2": {
            "sensitivity": float(recall_score(referable_true, referable_pred, zero_division=0)),
            "specificity": float(recall_score(referable_true, referable_pred, pos_label=0, zero_division=0)),
            "confusion_matrix": confusion_matrix(referable_true, referable_pred, labels=[0, 1]).tolist(),
        },
        "caution": "External research evaluation only; not clinical validation.",
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Saved report: {OUTPUT}")


if __name__ == "__main__":
    main()
