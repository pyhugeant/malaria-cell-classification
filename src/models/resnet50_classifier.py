import torch.nn as nn
from torchvision import models


class ResNet50Classifier(nn.Module):
    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        freeze_backbone=False,
    ):
        super().__init__()

        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        else:
            weights = None

        self.model = models.resnet50(weights=weights)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def build_model(cfg):
    model_name = cfg["model"]["name"].lower()

    if model_name == "resnet50":
        return ResNet50Classifier(
            num_classes=cfg["model"]["num_classes"],
            pretrained=cfg["model"]["pretrained"],
            freeze_backbone=cfg["model"]["freeze_backbone"],
        )

    raise ValueError(f"Unsupported model: {model_name}")