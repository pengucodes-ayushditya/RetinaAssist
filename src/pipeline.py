"""RetinaAssist: data loading, quality checks and preprocessing."""
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def quality_metrics(rgb: np.ndarray) -> dict:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 64:
        raise ValueError('Expected an RGB image of at least 64x64 pixels')

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    valid = gray > 12
    coverage = float(valid.mean())

    if coverage < .12:
        return {
            'usable': False,
            'reason': 'Insufficient illuminated retinal area',
            'warnings': [],
            'sharpness': 0.,
            'brightness': 0.,
            'coverage': coverage,
        }

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray[valid].mean())
    warnings = []

    if sharpness < 12.:
        warnings.append('Low sharpness')
    if brightness < 35.:
        warnings.append('Underexposed')
    if brightness > 225.:
        warnings.append('Overexposed')

    return {
        'usable': True,
        'reason': 'Retinal area detected',
        'warnings': warnings,
        'sharpness': sharpness,
        'brightness': brightness,
        'coverage': coverage,
    }


def crop_retina(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray > 10).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return rgb
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, h, _ = stats[idx]
    if w < 32 or h < 32:
        return rgb
    return rgb[y:y+h, x:x+w]


def tensor_transform(size: int = 224, train: bool = False):
    aug = [transforms.Resize((size, size))]
    if train:
        aug += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(12)]
    aug += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(aug)


def load_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def load_labels(csv: str | Path, image_root: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    # Native APTOS csv: id_code,diagnosis. Also supports image,diagnosis.
    if 'image' not in df and 'id_code' in df:
        df['image'] = df.id_code.astype(str) + '.png'
    if not {'image', 'diagnosis'}.issubset(df):
        raise ValueError('CSV needs image,diagnosis or id_code,diagnosis')
    df = df[['image', 'diagnosis']].dropna().copy()
    df['label'] = (pd.to_numeric(df.diagnosis) > 0).astype(int)
    root = Path(image_root)
    df['path'] = df.image.map(lambda x: str(root / str(x)))
    missing = df.loc[~df.path.map(lambda x: Path(x).is_file()), 'image'].tolist()
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} image(s), e.g. {missing[:3]}')
    return df.reset_index(drop=True)


class RetinaDataset(Dataset):
    def __init__(self, frame, train=False, size=224):
        self.frame = frame.reset_index(drop=True)
        self.transform = tensor_transform(size, train)

    def __len__(self): return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        rgb = crop_retina(load_rgb(row.path))
        x = self.transform(Image.fromarray(rgb))
        return x, torch.tensor(int(row.label), dtype=torch.long)
