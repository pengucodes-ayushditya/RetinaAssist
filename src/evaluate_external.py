import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.metrics import calculate_metrics
from src.model import load_model
from src.pipeline import RetinaDataset, load_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--report", default="outputs/idrid_external.json")
    args = parser.parse_args()

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    threshold = args.threshold if args.threshold is not None else saved.get("threshold", 0.5)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")

    frame = load_labels(args.csv, args.image_root)
    loader = DataLoader(
        RetinaDataset(frame, size=int(saved.get("size", 224))),
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model = load_model(args.checkpoint, device)
    labels, scores = [], []

    with torch.no_grad():
        for images, targets in loader:
            logits = model(images.to(device))
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
            labels.extend(targets.tolist())

    report = {
        "dataset": "IDRiD official testing split",
        "n_images": len(frame),
        "class_counts": frame.label.value_counts().sort_index().to_dict(),
        "threshold": threshold,
        "metrics": calculate_metrics(labels, scores, threshold=threshold),
        "checkpoint": args.checkpoint,
        "caution": "External research evaluation only; not clinical validation.",
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
