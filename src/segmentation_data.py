"""Patch-based IDRiD lesion and optic-disc segmentation data loader."""

from pathlib import Path
import argparse
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


CLASS_CODES = ("MA", "HE", "EX", "SE", "OD")
MASK_FOLDERS = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
    "OD": "5. Optic Disc",
}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class IDRiDSegmentationPatchDataset(Dataset):
    """Sample image patches and multi-label masks from IDRiD's training split.

    Missing class masks are marked invalid. They are not treated as
    background; the training loss must ignore those image/class pairs.
    """

    def __init__(
        self,
        root="data/external/idrid/A. Segmentation",
        patch_size=512,
        patches_per_image=8,
        image_ids=None,
        augment=True,
        deterministic=False,
        positive_patch_probability=0.7,
        seed=42,
    ):
        self.root = Path(root)
        self.image_dir = (
            self.root / "1. Original Images" / "a. Training Set"
        )
        self.mask_root = (
            self.root / "2. All Segmentation Groundtruths" / "a. Training Set"
        )
        self.patch_size = int(patch_size)
        self.patches_per_image = int(patches_per_image)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.positive_patch_probability = float(positive_patch_probability)
        self.rng = random.Random(self.seed)

        if self.patch_size <= 0 or self.patches_per_image <= 0:
            raise ValueError("patch_size and patches_per_image must be positive")
        if not 0.0 <= self.positive_patch_probability <= 1.0:
            raise ValueError("positive_patch_probability must be between 0 and 1")
        if not self.image_dir.is_dir() or not self.mask_root.is_dir():
            raise FileNotFoundError(
                "Could not find IDRiD segmentation training folders under "
                f"{self.root}"
            )

        for code, folder in MASK_FOLDERS.items():
            if not (self.mask_root / folder).is_dir():
                raise FileNotFoundError(
                    f"Missing mask folder for {code}: {self.mask_root / folder}"
                )

        image_paths = sorted(self.image_dir.glob("*.jpg"))
        if not image_paths:
            raise FileNotFoundError(f"No training JPG images found in {self.image_dir}")

        requested_ids = set(image_ids) if image_ids is not None else None
        self.records = []

        for image_path in image_paths:
            image_id = image_path.stem
            if requested_ids is not None and image_id not in requested_ids:
                continue

            masks = {
                code: self.mask_root / MASK_FOLDERS[code] / f"{image_id}_{code}.tif"
                for code in CLASS_CODES
            }
            self.records.append(
                {"image_id": image_id, "image_path": image_path, "mask_paths": masks}
            )

        if requested_ids is not None:
            found_ids = {record["image_id"] for record in self.records}
            missing_ids = requested_ids - found_ids
            if missing_ids:
                raise FileNotFoundError(
                    f"Training images not found for IDs: {sorted(missing_ids)}"
                )

        if not self.records:
            raise ValueError("No IDRiD training images matched the requested IDs")

    def __len__(self):
        return len(self.records) * self.patches_per_image

    def _load_record(self, record):
        with Image.open(record["image_path"]) as im:
            image = np.asarray(im.convert("RGB"))

        height, width = image.shape[:2]
        masks = np.zeros((len(CLASS_CODES), height, width), dtype=np.uint8)
        valid = np.zeros(len(CLASS_CODES), dtype=np.bool_)

        for channel, code in enumerate(CLASS_CODES):
            mask_path = record["mask_paths"][code]
            if not mask_path.is_file():
                continue

            with Image.open(mask_path) as im:
                mask = np.asarray(im.convert("L"))

            if mask.shape != (height, width):
                raise ValueError(
                    f"Image/mask size mismatch for {record['image_id']} {code}: "
                    f"image={(width, height)}, mask={(im.width, im.height)}"
                )

            masks[channel] = (mask > 0).astype(np.uint8)
            valid[channel] = True

        if height < self.patch_size or width < self.patch_size:
            raise ValueError(
                f"{record['image_id']} is smaller than patch_size={self.patch_size}"
            )

        return image, masks, valid

    def _choose_patch_origin(self, masks, valid, height, width, rng=None):
        rng = self.rng if rng is None else rng
        # Center some patches on every annotated class, including optic disc,
        # so small lesions and the larger structure are both sampled.
        annotated_classes = [
            channel
            for channel in range(len(CLASS_CODES))
            if valid[channel] and masks[channel].any()
        ]

        if (
            annotated_classes
            and rng.random() < self.positive_patch_probability
        ):
            channel = rng.choice(annotated_classes)
            yx = np.argwhere(masks[channel] > 0)
            y, x = yx[rng.randrange(len(yx))]
            top = min(max(int(y) - self.patch_size // 2, 0), height - self.patch_size)
            left = min(max(int(x) - self.patch_size // 2, 0), width - self.patch_size)
            return top, left

        top = rng.randint(0, height - self.patch_size)
        left = rng.randint(0, width - self.patch_size)
        return top, left

    def __getitem__(self, index):
        record_index = index // self.patches_per_image
        record = self.records[record_index]
        image, masks, valid = self._load_record(record)

        height, width = image.shape[:2]
        sample_rng = (
            random.Random(self.seed + int(index))
            if self.deterministic
            else self.rng
        )
        top, left = self._choose_patch_origin(
            masks, valid, height, width, rng=sample_rng
        )
        bottom = top + self.patch_size
        right = left + self.patch_size

        image = image[top:bottom, left:right]
        masks = masks[:, top:bottom, left:right]

        if self.augment:
            if sample_rng.random() < 0.5:
                image = np.flip(image, axis=1)
                masks = np.flip(masks, axis=2)
            if sample_rng.random() < 0.5:
                image = np.flip(image, axis=0)
                masks = np.flip(masks, axis=1)

            rotations = sample_rng.randrange(4)
            if rotations:
                image = np.rot90(image, rotations, axes=(0, 1))
                masks = np.rot90(masks, rotations, axes=(1, 2))

        image = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0
        image = (image - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        masks = np.ascontiguousarray(masks, dtype=np.float32)

        return (
            torch.from_numpy(image),
            torch.from_numpy(masks),
            torch.from_numpy(valid.copy()),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="data/external/idrid/A. Segmentation"
    )
    parser.add_argument("--patch-size", type=int, default=512)
    args = parser.parse_args()

    dataset = IDRiDSegmentationPatchDataset(
        root=args.root,
        patch_size=args.patch_size,
        patches_per_image=1,
        augment=False,
    )
    image, masks, valid = dataset[0]

    print(f"Images: {len(dataset.records)}")
    print(f"Sample: {dataset.records[0]['image_id']}")
    print(f"Image tensor: {tuple(image.shape)}")
    print(f"Mask tensor: {tuple(masks.shape)}")
    print(f"Classes: {CLASS_CODES}")
    print(f"Annotations available: {valid.tolist()}")
    print(f"Positive pixels by class: {masks.sum(dim=(1, 2)).tolist()}")


if __name__ == "__main__":
    main()
