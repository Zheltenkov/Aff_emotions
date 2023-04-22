import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNet50(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.num_classes = num_classes
        # Replace last layer
        self.model.fc = nn.Linear(100352, num_classes)
        self.attention = nn.MultiheadAttention(embed_dim=2048, num_heads=8)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = nn.functional.interpolate(x, size=(7, 7))
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)

        return x


