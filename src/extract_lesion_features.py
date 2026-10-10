import argparse
from pathlib import Path

import numpy as np
import torch

from src.grading_models import LESION_FEATURE_COUNT
from src.pipeline import load_labels, load_rgb
from src.segmentation_data import IMAGENET_MEAN, IMAGENET_STD
from src.segmentation_model import ResNet18UNet


TILE_SIZE = 512
THRESHOLD = 0.5


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def image_features(rgb, model, device, batch_size):
    height, width = rgb.shape[:2]
    pad_h = (-height) % TILE_SIZE
    pad_w = (-width) % TILE_SIZE
    padded = np.pad(
        rgb,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="reflect",
    )
    padded_h, padded_w = padded.shape[:2]

    positions = [
        (top, left)
        for top in range(0, padded_h, TILE_SIZE)
        for left in range(0, padded_w, TILE_SIZE)
    ]

    probability_sum = np.zeros(5, dtype=np.float64)
    positive_count = np.zeros(5, dtype=np.float64)
    maximum_probability = np.zeros(5, dtype=np.float64)
    retinal_pixel_count = 0

    mean = IMAGENET_MEAN[:, None, None]
    std = IMAGENET_STD[:, None, None]

    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start : start + batch_size]
        batch = []

        for top, left in batch_positions:
            patch = padded[top : top + TILE_SIZE, left : left + TILE_SIZE]
            patch = patch.transpose(2, 0, 1).astype(np.float32) / 255.0
            batch.append((patch - mean) / std)

        inputs = torch.from_numpy(np.stack(batch)).to(device)
        probabilities = torch.sigmoid(model(inputs)).cpu().numpy()

        for index, (top, left) in enumerate(batch_positions):
            real_h = min(TILE_SIZE, height - top)
            real_w = min(TILE_SIZE, width - left)
            if real_h <= 0 or real_w <= 0:
                continue

            rgb_tile = rgb[top : top + real_h, left : left + real_w]
            retinal = rgb_tile.mean(axis=2) > 12
            if not retinal.any():
                continue

            pixel_probabilities = probabilities[
                index, :, :real_h, :real_w
            ][:, retinal]

            retinal_pixel_count += int(retinal.sum())
            probability_sum += pixel_probabilities.sum(axis=1)
            positive_count += (pixel_probabilities >= THRESHOLD).sum(axis=1)
            maximum_probability = np.maximum(
                maximum_probability,
                pixel_probabilities.max(axis=1),
            )

    if retinal_pixel_count == 0:
        return np.zeros(LESION_FEATURE_COUNT, dtype=np.float32)

    area_fraction = positive_count / retinal_pixel_count
    mean_probability = probability_sum / retinal_pixel_count

    return np.concatenate(
        [area_fraction, mean_probability, maximum_probability]
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/labels.csv")
    parser.add_argument("--image-root", default="data/raw")
    parser.add_argument(
        "--checkpoint", default="models/idrid_segmentation_best.pt"
    )
    parser.add_argument(
        "--output", default="outputs/aptos_lesion_features.npz"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N images for a smoke test; 0 means all.",
    )
    args = parser.parse_args()

    device = choose_device()
    model = ResNet18UNet(pretrained=False, output_channels=5)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device).eval()

    frame = load_labels(args.csv, args.image_root)
    if args.limit > 0:
        frame = frame.head(args.limit)

    names = []
    features = []

    with torch.inference_mode():
        for index, row in enumerate(frame.itertuples(index=False), start=1):
            rgb = load_rgb(row.path)
            result = image_features(rgb, model, device, args.batch_size)
            if result.shape != (LESION_FEATURE_COUNT,):
                raise RuntimeError(f"Unexpected feature shape: {result.shape}")

            names.append(row.image)
            features.append(result)

            if index % 25 == 0 or index == len(frame):
                print(f"Processed {index}/{len(frame)} images", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        image_names=np.asarray(names),
        features=np.asarray(features, dtype=np.float32),
        feature_groups=np.asarray(
            ["area_fraction", "mean_probability", "maximum_probability"]
        ),
        threshold=np.asarray(THRESHOLD),
    )
    print(f"Feature matrix: ({len(features)}, {LESION_FEATURE_COUNT})")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
