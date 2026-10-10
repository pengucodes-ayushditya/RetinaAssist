"""ResNet-18 U-Net-style model for multi-label retinal segmentation."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, skip_channels, output_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                input_channels + skip_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.layers(torch.cat((x, skip), dim=1))


class ResNet18UNet(nn.Module):
    """Five-channel segmenter: MA, HE, EX, SE, and optic disc."""

    def __init__(self, pretrained=True, output_channels=5):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        self.pool = backbone.maxpool
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.decode3 = DecoderBlock(512, 256, 256)
        self.decode2 = DecoderBlock(256, 128, 128)
        self.decode1 = DecoderBlock(128, 64, 64)
        self.decode0 = DecoderBlock(64, 64, 64)

        self.refine = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(32, output_channels, kernel_size=1)

    def forward(self, image):
        input_size = image.shape[-2:]

        stem = self.stem(image)                         # H/2
        enc1 = self.encoder1(self.pool(stem))           # H/4
        enc2 = self.encoder2(enc1)                      # H/8
        enc3 = self.encoder3(enc2)                      # H/16
        enc4 = self.encoder4(enc3)                      # H/32

        x = self.decode3(enc4, enc3)
        x = self.decode2(x, enc2)
        x = self.decode1(x, enc1)
        x = self.decode0(x, stem)
        x = F.interpolate(
            x, size=input_size, mode="bilinear", align_corners=False
        )
        return self.classifier(self.refine(x))


if __name__ == "__main__":
    model = ResNet18UNet(pretrained=True)
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model = model.to(device).eval()

    with torch.no_grad():
        output = model(torch.zeros(1, 3, 512, 512, device=device))

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Parameters: {parameters:,}")
