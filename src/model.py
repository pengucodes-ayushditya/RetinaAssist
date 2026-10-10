import torch
from torchvision.models import resnet18, ResNet18_Weights


def make_model(pretrained: bool = False) -> torch.nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model


def load_model(checkpoint, device='cpu'):
    model = make_model(False)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state['model'])
    return model.to(device).eval()
