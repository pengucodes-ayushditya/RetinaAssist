import argparse
import json
import random
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

from src.grading_models import APTOSGradeClassifier
from src.pipeline import RetinaDataset, load_labels


GRADE_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative",
]


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
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    all_targets = []
    all_probabilities = []

    for images, targets in loader:
        logits = model(images.to(device))
        all_targets.extend(targets.tolist())
        all_probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy())

    targets = np.asarray(all_targets, dtype=np.int64)
    probabilities = np.asarray(all_probabilities)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/labels.csv")
    parser.add_argument("--image-root", default="data/raw")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out", default="models/aptos_grade5_best.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    frame = load_labels(args.csv, args.image_root)

    # Reproduce src.train's binary-label stratified split exactly.
    first, test = train_test_split(
        frame,
        test_size=0.15,
        stratify=frame.label,
        random_state=args.seed,
    )
    train, val = train_test_split(
        first,
        test_size=0.17647,
        stratify=first.label,
        random_state=args.seed,
    )

    split_ids = {
        "train": train.image.tolist(),
        "validation": val.image.tolist(),
        "test": test.image.tolist(),
    }

    # Preserve the same image split, but train on original APTOS grades 0–4.
    for split in (train, val):
        split["label"] = split["diagnosis"].astype(int)

    counts = np.bincount(train.label.to_numpy(), minlength=5)
    if np.any(counts == 0):
        raise ValueError(f"Training split is missing a grade: {counts.tolist()}")

    train_data = RetinaDataset(train, train=True, size=args.size)
    val_data = RetinaDataset(val, train=False, size=args.size)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = choose_device()
    model = APTOSGradeClassifier(pretrained=args.pretrained).to(device)

    # Moderate class balancing; the square root avoids extreme weights.
    class_weights = 1.0 / np.sqrt(counts.astype(np.float32))
    class_weights /= class_weights.mean()
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_qwk = -float("inf")
    stale_epochs = 0

    print(f"Device: {device}")
    print(
        f"Images: train={len(train)}, validation={len(val)}, "
        f"held-out test={len(test)}"
    )
    print(f"Training grade counts: {counts.tolist()}")
    print("Held-out test images are reserved; no test evaluation will run here.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_metrics = evaluate(model, val_loader, device)
        qwk = val_metrics["quadratic_weighted_kappa"]
        scheduler.step(qwk)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"loss={np.mean(losses):.4f} "
            f"val_qwk={qwk:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f}",
            flush=True,
        )

        if qwk > best_qwk:
            best_qwk = qwk
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": GRADE_NAMES,
                    "size": args.size,
                    "seed": args.seed,
                    "best_epoch": epoch,
                    "validation_metrics": val_metrics,
                    "split_ids": split_ids,
                },
                output,
            )
            print(f"Saved best checkpoint: {output}")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping.")
                break

    report_path = output.with_suffix(".validation.json")
    report_path.write_text(
        json.dumps(
            {
                "task": "APTOS five-grade classification",
                "train_count": len(train),
                "validation_count": len(val),
                "reserved_test_count": len(test),
                "best_validation_qwk": best_qwk,
                "checkpoint": str(output),
                "test_evaluated": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved validation summary: {report_path}")


if __name__ == "__main__":
    main()
