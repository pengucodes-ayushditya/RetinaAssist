import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from src.segmentation_data import (
    CLASS_CODES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MASK_FOLDERS,
)
from src.segmentation_model import ResNet18UNet


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def load_masks(mask_root, image_id, height, width):
    masks = np.zeros((len(CLASS_CODES), height, width), dtype=np.uint8)
    valid = np.zeros(len(CLASS_CODES), dtype=bool)

    for channel, code in enumerate(CLASS_CODES):
        mask_path = mask_root / MASK_FOLDERS[code] / f"{image_id}_{code}.tif"
        if not mask_path.is_file():
            continue

        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L"))

        if mask.shape != (height, width):
            raise ValueError(
                f"Image/mask size mismatch for {image_id} {code}: "
                f"image={(height, width)}, mask={mask.shape}"
            )

        masks[channel] = mask > 0
        valid[channel] = True

    return masks, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/external/idrid/A. Segmentation")
    parser.add_argument(
        "--checkpoint", default="models/idrid_segmentation_best.pt"
    )
    parser.add_argument(
        "--output", default="outputs/idrid_segmentation_test.json"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    root = Path(args.root)
    image_dir = root / "1. Original Images" / "b. Testing Set"
    mask_root = root / "2. All Segmentation Groundtruths" / "b. Testing Set"
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No test JPG images found in {image_dir}")

    device = choose_device()
    model = ResNet18UNet(pretrained=False, output_channels=len(CLASS_CODES))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device).eval()

    totals = {
        code: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "annotated_images": 0,
        }
        for code in CLASS_CODES
    }

    tile_size = 512
    mean = IMAGENET_MEAN[:, None, None]
    std = IMAGENET_STD[:, None, None]

    with torch.inference_mode():
        for image_path in image_paths:
            image_id = image_path.stem
            rgb = load_image(image_path)
            height, width = rgb.shape[:2]
            masks, valid = load_masks(mask_root, image_id, height, width)

            pad_h = (-height) % tile_size
            pad_w = (-width) % tile_size
            padded = np.pad(
                rgb,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="reflect",
            )
            padded_h, padded_w = padded.shape[:2]

            positions = [
                (top, left)
                for top in range(0, padded_h, tile_size)
                for left in range(0, padded_w, tile_size)
            ]

            for start in range(0, len(positions), args.batch_size):
                batch_positions = positions[start : start + args.batch_size]
                batch = []

                for top, left in batch_positions:
                    patch = padded[
                        top : top + tile_size,
                        left : left + tile_size,
                    ]
                    patch = patch.transpose(2, 0, 1).astype(np.float32) / 255.0
                    patch = (patch - mean) / std
                    batch.append(patch)

                inputs = torch.from_numpy(np.stack(batch)).to(device)
                probabilities = torch.sigmoid(model(inputs)).cpu().numpy()
                predictions = probabilities >= args.threshold

                for index, (top, left) in enumerate(batch_positions):
                    real_h = min(tile_size, height - top)
                    real_w = min(tile_size, width - left)
                    if real_h <= 0 or real_w <= 0:
                        continue

                    for channel, code in enumerate(CLASS_CODES):
                        if not valid[channel]:
                            continue

                        predicted = predictions[
                            index, channel, :real_h, :real_w
                        ]
                        target = masks[
                            channel,
                            top : top + real_h,
                            left : left + real_w,
                        ].astype(bool)

                        totals[code]["tp"] += int(np.count_nonzero(predicted & target))
                        totals[code]["fp"] += int(np.count_nonzero(predicted & ~target))
                        totals[code]["fn"] += int(np.count_nonzero(~predicted & target))

            for channel, code in enumerate(CLASS_CODES):
                if valid[channel]:
                    totals[code]["annotated_images"] += 1

            print(f"Evaluated {image_id}")

    per_class = {}
    for code, counts in totals.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        dice_denominator = 2 * tp + fp + fn
        iou_denominator = tp + fp + fn

        per_class[code] = {
            "annotated_images": counts["annotated_images"],
            "dice": (2 * tp / dice_denominator) if dice_denominator else None,
            "iou": (tp / iou_denominator) if iou_denominator else None,
            "precision": (tp / (tp + fp)) if (tp + fp) else None,
            "sensitivity": (tp / (tp + fn)) if (tp + fn) else None,
        }

    valid_dice = [
        values["dice"]
        for values in per_class.values()
        if values["dice"] is not None
    ]
    report = {
        "dataset": "IDRiD official segmentation testing split",
        "n_images": len(image_paths),
        "threshold": args.threshold,
        "macro_dice": sum(valid_dice) / len(valid_dice) if valid_dice else None,
        "per_class": per_class,
        "checkpoint": args.checkpoint,
        "caution": "Research evaluation only; not clinical validation.",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
