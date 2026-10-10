"""Train five-grade DR classifier on APTOS train + official IDRiD train."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader

from src.grading_models import APTOSGradeClassifier
from src.pipeline import RetinaDataset, load_labels


def image_key(value):
    return Path(str(value)).stem


def evaluate(model, loader, device):
    model.eval()
    true, predicted = [], []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
            true.extend(targets.tolist())

    return {
        "n": len(true),
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(true, predicted, weights="quadratic")
        ),
    }


def load_idrid_training(csv_path, image_root):
    labels = pd.read_csv(csv_path)
    labels.columns = labels.columns.str.strip()

    name_col = "Image name"
    grade_col = "Retinopathy grade"
    if name_col not in labels or grade_col not in labels:
        raise ValueError(f"Unexpected IDRiD CSV columns: {list(labels.columns)}")

    image_by_stem = {
        p.stem: p for p in Path(image_root).rglob("*") if p.is_file()
    }
    rows, missing = [], []

    for _, row in labels.iterrows():
        name = str(row[name_col]).strip()
        image_path = image_by_stem.get(name)
        if image_path is None:
            missing.append(name)
            continue
        rows.append({
            "image": name,
            "path": str(image_path),
            "label": int(row[grade_col]),
        })

    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} IDRiD training images, e.g. {missing[:5]}"
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aptos-csv", default="data/labels.csv")
    parser.add_argument("--aptos-image-root", default="data/raw")
    parser.add_argument(
        "--idrid-csv",
        default="data/external/idrid/2. Groundtruths/"
                "a. IDRiD_Disease Grading_Training Labels.csv",
    )
    parser.add_argument(
        "--idrid-image-root",
        default="data/external/idrid/1. Original Images/a. Training Set",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--out", default="models/aptos_idrid_grade5_best.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    aptos = load_labels(args.aptos_csv, args.aptos_image_root)
    aptos["grade"] = pd.to_numeric(aptos["diagnosis"]).astype(int)

    # Preserve the APTOS seed-42 stratified split proportions used previously.
    first, aptos_test = train_test_split(
        aptos, test_size=0.15, stratify=aptos["label"], random_state=args.seed
    )
    aptos_train, aptos_val = train_test_split(
        first, test_size=0.17647, stratify=first["label"], random_state=args.seed
    )

    aptos_train = aptos_train.copy()
    aptos_val = aptos_val.copy()
    aptos_train["label"] = aptos_train["grade"]
    aptos_val["label"] = aptos_val["grade"]

    idrid_train = load_idrid_training(args.idrid_csv, args.idrid_image_root)
    train_frame = pd.concat(
        [
            aptos_train[["image", "path", "label"]],
            idrid_train[["image", "path", "label"]],
        ],
        ignore_index=True,
    )

    train_counts = (
        train_frame["label"].value_counts().reindex(range(5), fill_value=0).sort_index()
    )
    print(f"APTOS: train={len(aptos_train)}, validation={len(aptos_val)}, "
          f"held-out test reserved={len(aptos_test)}")
    print(f"IDRiD official training images: {len(idrid_train)}")
    print(f"Combined training grade counts: {train_counts.tolist()}")

    train_dataset = ConcatDataset([
        RetinaDataset(train_frame, train=True, size=args.size),
    ])
    val_dataset = RetinaDataset(aptos_val, train=False, size=args.size)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    device = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    model = APTOSGradeClassifier(pretrained=args.pretrained).to(device)
    counts = torch.tensor(train_counts.to_numpy(), dtype=torch.float32, device=device)
    class_weights = counts.clamp_min(1).rsqrt()
    class_weights = class_weights / class_weights.mean()
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_qwk = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_metrics = evaluate(model, val_loader, device)
        qwk = val_metrics["quadratic_weighted_kappa"]
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"loss={np.mean(losses):.4f} "
            f"val_qwk={qwk:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} "
            f"val_accuracy={val_metrics['accuracy']:.4f}",
            flush=True,
        )

        if qwk > best_qwk:
            best_qwk = qwk
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": ["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
                    "size": args.size,
                    "seed": args.seed,
                    "best_epoch": best_epoch,
                    "validation_metrics": val_metrics,
                    "training_sources": ["APTOS train", "IDRiD official train"],
                    "split_ids": {
                        "aptos_train": aptos_train["image"].astype(str).tolist(),
                        "aptos_validation": aptos_val["image"].astype(str).tolist(),
                        "aptos_test_reserved": aptos_test["image"].astype(str).tolist(),
                        "idrid_train": idrid_train["image"].astype(str).tolist(),
                    },
                },
                output,
            )
            print(f"Saved best checkpoint: {output}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping.")
                break

    print(f"Best validation QWK: {best_qwk:.4f} at epoch {best_epoch}")
    print("APTOS held-out test and IDRiD official test were not evaluated during training.")


if __name__ == "__main__":
    main()
