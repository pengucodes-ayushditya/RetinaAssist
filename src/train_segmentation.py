import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.segmentation_data import CLASS_CODES, IDRiDSegmentationPatchDataset
from src.segmentation_model import ResNet18UNet


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def masked_loss(logits, masks, valid):
    # valid has shape [batch, classes]; missing annotations are ignored.
    valid_pixels = valid[:, :, None, None].to(logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, masks, reduction="none")
    bce = (bce * valid_pixels).sum()
    bce /= (valid_pixels.sum() * masks.shape[-2] * masks.shape[-1]).clamp_min(1)

    probs = torch.sigmoid(logits)
    dims = (-2, -1)
    intersection = (probs * masks).sum(dim=dims)
    denominator = probs.sum(dim=dims) + masks.sum(dim=dims)
    dice = (2 * intersection + 1) / (denominator + 1)
    dice_loss = ((1 - dice) * valid.to(dice.dtype)).sum()
    dice_loss /= valid.sum().clamp_min(1)

    return bce + dice_loss


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    loss_total = 0.0
    batches = 0
    classes = len(CLASS_CODES)
    intersection = torch.zeros(classes, dtype=torch.float64)
    pred_total = torch.zeros(classes, dtype=torch.float64)
    target_total = torch.zeros(classes, dtype=torch.float64)
    annotated = torch.zeros(classes, dtype=torch.float64)

    for images, masks, valid in loader:
        images = images.to(device)
        masks = masks.to(device)
        valid = valid.to(device).bool()

        logits = model(images)
        loss_total += masked_loss(logits, masks, valid).item()
        batches += 1

        predictions = (torch.sigmoid(logits) >= 0.5).to(masks.dtype)
        for class_idx in range(classes):
            selected = valid[:, class_idx]
            if selected.any():
                p = predictions[selected, class_idx]
                y = masks[selected, class_idx]
                intersection[class_idx] += (p * y).sum().cpu().double()
                pred_total[class_idx] += p.sum().cpu().double()
                target_total[class_idx] += y.sum().cpu().double()
                annotated[class_idx] += selected.sum().cpu().double()

    per_class_dice = {}
    for i, name in enumerate(CLASS_CODES):
        if annotated[i] > 0:
            per_class_dice[name] = (
                (2 * intersection[i] + 1) / (pred_total[i] + target_total[i] + 1)
            ).item()

    macro_dice = sum(per_class_dice.values()) / max(len(per_class_dice), 1)
    return loss_total / max(batches, 1), macro_dice, per_class_dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/external/idrid/A. Segmentation")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patches-per-image", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--output", default="models/idrid_segmentation_best.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.root)
    train_images = root / "1. Original Images" / "a. Training Set"
    image_ids = sorted(path.stem for path in train_images.glob("*.jpg"))
    if len(image_ids) < 2:
        raise RuntimeError(f"Expected IDRiD training images under {train_images}")

    rng = random.Random(args.seed)
    rng.shuffle(image_ids)
    val_count = max(1, round(len(image_ids) * 0.2))
    val_ids = sorted(image_ids[:val_count])
    train_ids = sorted(image_ids[val_count:])

    train_data = IDRiDSegmentationPatchDataset(
        root=str(root),
        image_ids=train_ids,
        patches_per_image=args.patches_per_image,
        augment=True,
        positive_patch_probability=0.7,
        seed=args.seed,
    )
    val_data = IDRiDSegmentationPatchDataset(
        root=str(root),
        image_ids=val_ids,
        patches_per_image=args.patches_per_image,
        augment=False,
        deterministic=True,
        positive_patch_probability=0.5,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    device = choose_device()
    model = ResNet18UNet(pretrained=True, output_channels=len(CLASS_CODES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_dice = -1.0
    epochs_without_improvement = 0

    print(f"Device: {device}")
    print(f"Training images: {len(train_ids)}; validation images: {len(val_ids)}")
    print(f"Validation IDs: {', '.join(val_ids)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_batches = 0

        for images, masks, valid in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            valid = valid.to(device).bool()

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = masked_loss(logits, masks, valid)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            train_batches += 1

        val_loss, val_dice, per_class = validate(model, val_loader, device)
        scheduler.step(val_dice)
        train_loss = train_loss_total / max(train_batches, 1)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_dice={val_dice:.4f} "
            f"per_class={json.dumps(per_class)}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASS_CODES,
                    "validation_dice": best_dice,
                    "train_image_ids": train_ids,
                    "validation_image_ids": val_ids,
                    "patch_size": 512,
                },
                output_path,
            )
            print(f"Saved best checkpoint: {output_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping.")
                break


if __name__ == "__main__":
    main()
