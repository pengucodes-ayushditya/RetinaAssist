import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.grading_models import (
    APTOSGradeClassifier,
    LESION_FEATURE_COUNT,
    LesionFusionGradeClassifier,
)
from src.pipeline import RetinaDataset, load_labels


GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


class FusionTestDataset(RetinaDataset):
    def __init__(self, frame, feature_map, size=224):
        super().__init__(frame, train=False, size=size)
        self.feature_map = feature_map

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        name = str(self.frame.iloc[index].image)
        features = torch.from_numpy(
            self.feature_map[name].astype(np.float32, copy=False)
        )
        return image, target, features


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def binary_metrics(targets, scores, threshold=0.5):
    predictions = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(
        targets, predictions, labels=[0, 1]
    ).ravel()
    return {
        "threshold": threshold,
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


@torch.inference_mode()
def predict(model, loader, device, fusion=False):
    model.eval()
    targets_all = []
    probabilities_all = []

    for batch in loader:
        if fusion:
            images, targets, features = batch
            logits = model(images.to(device), features.to(device))
        else:
            images, targets = batch
            logits = model(images.to(device))

        targets_all.extend(targets.tolist())
        probabilities_all.extend(torch.softmax(logits, dim=1).cpu().numpy())

    return np.asarray(targets_all, dtype=np.int64), np.asarray(probabilities_all)


def summarize(targets, probabilities):
    predictions = probabilities.argmax(axis=1)
    return {
        "n": int(len(targets)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=list(range(5)),
                average="macro",
                zero_division=0,
            )
        ),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(targets, predictions, weights="quadratic")
        ),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=list(range(5))
        ).tolist(),
        "per_grade_f1": {
            GRADE_NAMES[i]: float(
                f1_score(
                    targets == i,
                    predictions == i,
                    zero_division=0,
                )
            )
            for i in range(5)
        },
        "any_dr": binary_metrics(
            (targets > 0).astype(int),
            probabilities[:, 1:].sum(axis=1),
        ),
        "referable_dr_grade_ge_2": binary_metrics(
            (targets >= 2).astype(int),
            probabilities[:, 2:].sum(axis=1),
        ),
    }


def load_project_checkpoint(path, device):
    # These are checkpoints created locally by this project's training scripts.
    return torch.load(path, map_location=device, weights_only=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/labels.csv")
    parser.add_argument("--image-root", default="data/raw")
    parser.add_argument(
        "--features", default="outputs/aptos_lesion_features.npz"
    )
    parser.add_argument(
        "--baseline", default="models/aptos_grade5_best.pt"
    )
    parser.add_argument(
        "--fusion", default="models/aptos_grade5_fusion_best.pt"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--output", default="outputs/aptos_grade5_comparison.json"
    )
    args = parser.parse_args()

    device = choose_device()
    frame = load_labels(args.csv, args.image_root)

    # Match the existing seed-42 stratified train/validation/test split.
    first, test = train_test_split(
        frame,
        test_size=0.15,
        stratify=frame.label,
        random_state=args.seed,
    )
    _, _ = train_test_split(
        first,
        test_size=0.17647,
        stratify=first.label,
        random_state=args.seed,
    )

    baseline_ckpt = load_project_checkpoint(args.baseline, device)
    fusion_ckpt = load_project_checkpoint(args.fusion, device)

    # Verify both models were trained with this same partition.
    expected_test_ids = set(test.image.tolist())
    for name, checkpoint in (
        ("baseline", baseline_ckpt),
        ("fusion", fusion_ckpt),
    ):
        saved_ids = checkpoint.get("split_ids", {}).get("test")
        if saved_ids is None or set(saved_ids) != expected_test_ids:
            raise ValueError(
                f"{name} checkpoint test IDs do not match the seed-{args.seed} split"
            )

    test = test.copy()
    test["label"] = test["diagnosis"].astype(int)
    test_loader = DataLoader(
        RetinaDataset(test, train=False, size=args.size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    baseline_model = APTOSGradeClassifier(pretrained=False).to(device)
    baseline_model.load_state_dict(baseline_ckpt["model_state_dict"])
    baseline_targets, baseline_probs = predict(
        baseline_model, test_loader, device, fusion=False
    )

    feature_file = np.load(args.features, allow_pickle=False)
    feature_names = [str(name) for name in feature_file["image_names"]]
    feature_values = feature_file["features"]
    if feature_values.shape != (len(feature_names), LESION_FEATURE_COUNT):
        raise ValueError("Lesion feature cache has an unexpected shape")
    feature_map = dict(zip(feature_names, feature_values))
    missing = sorted(set(test.image) - set(feature_map))
    if missing:
        raise ValueError(f"Missing lesion features for test images: {missing[:3]}")

    fusion_loader = DataLoader(
        FusionTestDataset(test, feature_map, size=args.size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    fusion_model = LesionFusionGradeClassifier(pretrained=False).to(device)
    fusion_model.load_state_dict(fusion_ckpt["model_state_dict"])
    fusion_targets, fusion_probs = predict(
        fusion_model, fusion_loader, device, fusion=True
    )

    if not np.array_equal(baseline_targets, fusion_targets):
        raise RuntimeError("Baseline and fusion test targets do not match")

    report = {
        "dataset": "APTOS held-out test split",
        "n_images": int(len(test)),
        "seed": args.seed,
        "binary_threshold": 0.5,
        "baseline": {
            "best_epoch": int(baseline_ckpt["best_epoch"]),
            "validation_qwk": float(
                baseline_ckpt["validation_metrics"][
                    "quadratic_weighted_kappa"
                ]
            ),
            "test_metrics": summarize(baseline_targets, baseline_probs),
        },
        "lesion_fusion": {
            "best_epoch": int(fusion_ckpt["best_epoch"]),
            "validation_qwk": float(
                fusion_ckpt["validation_metrics"][
                    "quadratic_weighted_kappa"
                ]
            ),
            "test_metrics": summarize(fusion_targets, fusion_probs),
        },
        "caution": "Research evaluation only; not clinical validation.",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved comparison: {output}")


if __name__ == "__main__":
    main()
