import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, recall_score
from torch.utils.data import DataLoader

from src.grading_models import APTOSGradeClassifier
from src.pipeline import RetinaDataset, load_labels


CHECKPOINT = Path("models/aptos_idrid_grade5_best.pt")
OUTPUT = Path("outputs/aptos_idrid_grade5_thresholds.json")


def image_key(value):
    return Path(str(value)).stem


def choose_threshold(y_true, scores, minimum_sensitivity):
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    valid = [
        float(t) for t in candidates
        if recall_score(y_true, scores >= t, zero_division=0) >= minimum_sensitivity
    ]
    return max(valid) if valid else 0.0


def summarize(y_true, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    sensitivity = matrix[1, 1] / max(1, matrix[1].sum())
    specificity = matrix[0, 0] / max(1, matrix[0].sum())
    return {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "confusion_matrix": matrix.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Select screening thresholds using an APTOS validation split."
    )
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--csv", default="data/labels.csv")
    parser.add_argument("--image-root", default="data/raw")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    labels = load_labels(args.csv, args.image_root)
    labels["grade"] = pd.to_numeric(labels["diagnosis"]).astype(int)
    labels["image_key"] = labels["image"].map(image_key)

    validation_ids = {image_key(x) for x in checkpoint["split_ids"].get("validation", checkpoint["split_ids"].get("aptos_validation", []))}
    validation = labels[labels["image_key"].isin(validation_ids)].copy()
    if len(validation) != len(validation_ids):
        raise ValueError(
            f"Validation ID mismatch: checkpoint has {len(validation_ids)} IDs, "
            f"matched {len(validation)} rows."
        )

    validation["label"] = validation["grade"]
    dataset = RetinaDataset(
        validation, train=False, size=int(checkpoint.get("size", 224))
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = APTOSGradeClassifier(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    grades, probabilities = [], []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            grades.extend(targets.numpy().tolist())

    grades = np.asarray(grades, dtype=int)
    probabilities = np.concatenate(probabilities)
    any_dr_scores = 1.0 - probabilities[:, 0]
    referable_scores = probabilities[:, 2:].sum(axis=1)

    any_dr_true = (grades >= 1).astype(int)
    referable_true = (grades >= 2).astype(int)

    # Choose the highest threshold that still meets each sensitivity target.
    any_dr_threshold = choose_threshold(any_dr_true, any_dr_scores, 0.95)
    referable_threshold = choose_threshold(referable_true, referable_scores, 0.90)

    result = {
        "source": "APTOS validation split",
        "n_images": int(len(grades)),
        "selection_rule": {
            "any_dr_minimum_sensitivity": 0.95,
            "referable_dr_minimum_sensitivity": 0.90,
            "rule": "Highest threshold meeting the validation sensitivity target",
        },
        "any_dr": summarize(any_dr_true, any_dr_scores, any_dr_threshold),
        "referable_dr_grade_ge_2": summarize(
            referable_true, referable_scores, referable_threshold
        ),
        "caution": "Thresholds selected on APTOS validation; IDRiD test labels were not used.",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved thresholds: {output_path}")


if __name__ == "__main__":
    main()
