"""Inference helpers for the RetinaAssist research demo."""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.grading_models import APTOSGradeClassifier
from src.pipeline import crop_retina, quality_metrics, tensor_transform
from src.segmentation_data import CLASS_CODES, IMAGENET_MEAN, IMAGENET_STD
from src.segmentation_model import ResNet18UNet


GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
SEGMENTATION_COLORS = {
    "MA": (255, 45, 85),
    "HE": (255, 145, 45),
    "EX": (255, 220, 45),
    "SE": (35, 205, 220),
    "OD": (180, 100, 255),
}


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_grade5_model(checkpoint_path, device=None):
    device = device or choose_device()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = APTOSGradeClassifier(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


def load_segmentation_model(checkpoint_path, device=None):
    device = device or choose_device()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = ResNet18UNet(pretrained=False, output_channels=len(CLASS_CODES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


def predict_grade5(rgb, model, size=224, thresholds_path=None):
    quality = quality_metrics(rgb)
    result = {"quality": quality, "status": "screened"}

    if not quality["usable"]:
        result.update({
            "status": "quality_review",
            "message": "Image quality is insufficient for an automated demo result.",
        })
        return result

    device = next(model.parameters()).device
    cropped = crop_retina(rgb)
    image = Image.fromarray(cropped)
    tensor = tensor_transform(size, train=False)(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    any_dr_score = float(1.0 - probabilities[0])
    referable_score = float(probabilities[2:].sum())
    thresholds = {"any_dr": 0.5, "referable_dr_grade_ge_2": 0.5}
    threshold_source = "Default 0.5 thresholds"

    if thresholds_path and Path(thresholds_path).is_file():
        import json
        saved = json.loads(Path(thresholds_path).read_text())
        thresholds = {
            "any_dr": float(saved["any_dr"]["threshold"]),
            "referable_dr_grade_ge_2": float(
                saved["referable_dr_grade_ge_2"]["threshold"]
            ),
        }
        threshold_source = "APTOS validation split"

    grade_index = int(np.argmax(probabilities))
    result.update({
        "grade_index": grade_index,
        "grade_name": GRADE_NAMES[grade_index],
        "grade_score": float(probabilities[grade_index]),
        "grade_probabilities": {
            name: float(probabilities[i]) for i, name in enumerate(GRADE_NAMES)
        },
        "any_dr_score": any_dr_score,
        "referable_score": referable_score,
        "thresholds": thresholds,
        "threshold_source": threshold_source,
        "any_dr_signal": any_dr_score >= thresholds["any_dr"],
        "referable_dr_signal": (
            referable_score >= thresholds["referable_dr_grade_ge_2"]
        ),
    })
    return result


def predict_segmentation(rgb, model, threshold=0.5, batch_size=4):
    """Tile a full-resolution RGB image and return predicted masks and overlay."""
    height, width = rgb.shape[:2]
    tile_size = 512
    pad_h = (-height) % tile_size
    pad_w = (-width) % tile_size
    padded = np.pad(
        rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"
    )
    padded_h, padded_w = padded.shape[:2]
    positions = [
        (top, left)
        for top in range(0, padded_h, tile_size)
        for left in range(0, padded_w, tile_size)
    ]

    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    masks = np.zeros((len(CLASS_CODES), height, width), dtype=bool)
    device = next(model.parameters()).device

    with torch.inference_mode():
        for start in range(0, len(positions), batch_size):
            batch_positions = positions[start:start + batch_size]
            patches = []
            for top, left in batch_positions:
                patch = padded[top:top + tile_size, left:left + tile_size]
                patch = patch.astype(np.float32) / 255.0
                patch = (patch - mean) / std
                patches.append(patch.transpose(2, 0, 1))

            inputs = torch.from_numpy(np.stack(patches)).to(device)
            probabilities = torch.sigmoid(model(inputs)).cpu().numpy()
            predictions = probabilities >= threshold

            for index, (top, left) in enumerate(batch_positions):
                real_h = min(tile_size, height - top)
                real_w = min(tile_size, width - left)
                if real_h <= 0 or real_w <= 0:
                    continue
                masks[:, top:top + real_h, left:left + real_w] = (
                    predictions[index, :, :real_h, :real_w]
                )

    color_layer = np.zeros(rgb.shape, dtype=np.float32)
    color_count = np.zeros((height, width, 1), dtype=np.float32)
    area_percentages = {}

    for channel, code in enumerate(CLASS_CODES):
        mask = masks[channel]
        area_percentages[code] = float(mask.mean() * 100.0)
        color = np.asarray(SEGMENTATION_COLORS[code], dtype=np.float32)
        color_layer += mask[..., None] * color
        color_count += mask[..., None]

    covered = color_count[..., 0] > 0
    color_layer[covered] /= color_count[covered]
    overlay = rgb.astype(np.float32).copy()
    overlay[covered] = 0.6 * overlay[covered] + 0.4 * color_layer[covered]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return {
        "overlay": overlay,
        "area_percentages": area_percentages,
        "threshold": float(threshold),
    }
