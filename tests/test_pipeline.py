import numpy as np
import pytest
from src.pipeline import quality_metrics, crop_retina
from src.metrics import calculate_metrics


def test_dark_image_rejected():
    img=np.zeros((128,128,3),dtype=np.uint8)
    assert not quality_metrics(img)['usable']


def test_crop_preserves_rgb():
    img=np.zeros((128,128,3),dtype=np.uint8)
    img[20:100,20:100]=125
    crop=crop_retina(img)
    assert crop.ndim==3 and crop.shape[-1]==3 and crop.shape[0]<=128


def test_metrics():
    m=calculate_metrics([0,0,1,1],[.1,.9,.8,.9])
    assert m['confusion_matrix']==[[1,1],[0,2]]
    assert m['sensitivity']==1


def test_small_image_rejected():
    with pytest.raises(ValueError):quality_metrics(np.zeros((32,32,3),dtype=np.uint8))


def test_low_sharpness_is_warning_not_rejection():
    image = np.full((128, 128, 3), 100, dtype=np.uint8)
    result = quality_metrics(image)
    assert result["usable"] is True
    assert "Low sharpness" in result["warnings"]
