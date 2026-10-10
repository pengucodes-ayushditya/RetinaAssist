"""Grad-CAM for RetinaAssist ResNet18 classifiers."""
import cv2
import numpy as np
import torch
from PIL import Image

from src.pipeline import crop_retina, tensor_transform


def gradcam_overlay(model, rgb: np.ndarray, class_index=None):
    cropped = crop_retina(rgb)
    device = next(model.parameters()).device
    image = tensor_transform(224, train=False)(
        Image.fromarray(cropped)
    ).unsqueeze(0).to(device)

    encoder = getattr(model, "encoder", model)
    target_layer = encoder.layer4[-1]
    activations, gradients = {}, {}

    forward_hook = target_layer.register_forward_hook(
        lambda module, inputs, output: activations.update(value=output)
    )
    backward_hook = target_layer.register_full_backward_hook(
        lambda module, grad_input, grad_output: gradients.update(value=grad_output[0])
    )

    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(image)
            selected_class = (
                int(logits.argmax(dim=1).item())
                if class_index is None
                else int(class_index)
            )
            logits[0, selected_class].backward()

        weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu(
            (weights * activations["value"]).sum(dim=1)
        )[0].detach().cpu().numpy()

        cam = cv2.resize(cam, (cropped.shape[1], cropped.shape[0]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(cropped, 0.6, heat, 0.4, 0)
    finally:
        forward_hook.remove()
        backward_hook.remove()
