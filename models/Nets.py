import torch
import torch.nn as nn
import torchvision.models as models

# ResNet18 backbone for medical image classification
class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Gradient Score Network (GSN) for support-aware gradient sparsification
class GradientScoreNetwork(nn.Module):
    def __init__(self, input_dim):
        super(GradientScoreNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()  # Output probability for each gradient coordinate
        )

    def forward(self, x):
        return self.mlp(x)