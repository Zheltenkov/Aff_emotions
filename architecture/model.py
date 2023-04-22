import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.conv = nn.Conv2d(input_dim, 1, kernel_size=1, bias=False)

    def forward(self, x):
        weights = F.relu(self.conv(x))
        weights = torch.softmax(weights.view(weights.size(0), -1), dim=1)
        weights = weights.view(weights.size(0), 1, x.size(2), x.size(3))
        return weights * x


class AffModelClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(AffModelClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.att1 = Attention(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.att2 = Attention(128)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.att3 = Attention(128)
        #64x64
        self.fc1 = nn.Linear(8 * 64 * 64, 512, bias=False)
        # 256x256
        #self.fc1 = nn.Linear(128 * 64 * 64, 512, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.att1(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.att2(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.att3(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        #return F.softmax(x, dim=1)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=3)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        weights = self.conv1x1(x)
        weights = self.softmax(weights)
        x = x * weights
        x = x.sum(dim=2)
        return x


class AffModelRegressor(nn.Module):
    def __init__(self, n_channels=3):
        super(AffModelRegressor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1)
        self.attention1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1), nn.Sigmoid())
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.attention2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1), nn.Sigmoid())
        self.fc = nn.Linear(128 * 64 * 64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        att1 = self.attention1(x)
        x = x * att1
        x = F.relu(self.conv2(x))
        att2 = self.attention2(x)
        x = x * att2
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc(x))
        return x







