import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


NUM_GRADES = 5
LESION_FEATURE_COUNT = 15


def make_encoder(pretrained=False):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    encoder = resnet18(weights=weights)
    feature_count = encoder.fc.in_features
    encoder.fc = nn.Identity()
    return encoder, feature_count


class APTOSGradeClassifier(nn.Module):
    """Image-only ResNet18 classifier for APTOS grades 0 through 4."""

    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder, feature_count = make_encoder(pretrained)
        self.classifier = nn.Linear(feature_count, NUM_GRADES)

    def forward(self, images):
        features = self.encoder(images)
        return self.classifier(features)


class LesionFusionGradeClassifier(nn.Module):
    """ResNet18 image features fused with 15 segmentation-summary features."""

    def __init__(self, pretrained=False):
        super().__init__()
        self.encoder, feature_count = make_encoder(pretrained)
        self.classifier = nn.Sequential(
            nn.Linear(feature_count + LESION_FEATURE_COUNT, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, NUM_GRADES),
        )

    def forward(self, images, lesion_features):
        image_features = self.encoder(images)
        fused = torch.cat((image_features, lesion_features), dim=1)
        return self.classifier(fused)


if __name__ == "__main__":
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    image = torch.zeros(1, 3, 224, 224, device=device)
    lesion_features = torch.zeros(
        1, LESION_FEATURE_COUNT, device=device
    )

    baseline = APTOSGradeClassifier(pretrained=False).to(device).eval()
    fusion = LesionFusionGradeClassifier(pretrained=False).to(device).eval()

    with torch.inference_mode():
        print("Device:", device)
        print("Baseline output:", tuple(baseline(image).shape))
        print("Fusion output:", tuple(fusion(image, lesion_features).shape))
        print(
            "Baseline parameters:",
            sum(p.numel() for p in baseline.parameters()),
        )
        print(
            "Fusion parameters:",
            sum(p.numel() for p in fusion.parameters()),
        )
