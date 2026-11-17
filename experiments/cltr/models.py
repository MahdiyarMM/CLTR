"""CIFAR ResNet-32 (3 stages x 5 basic blocks, option-A shortcuts)."""

import torch.nn as nn
import torch.nn.functional as F


class Shortcut(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.pad = planes // 4

    def forward(self, x):
        return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.pad, self.pad))


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = Shortcut(planes) if stride != 1 or in_planes != planes else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class ResNet32(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 1)
        self.layer2 = self._make_layer(32, 2)
        self.layer3 = self._make_layer(64, 2)
        self.fc = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)

    def _make_layer(self, planes, stride):
        layers = []
        for s in [stride] + [1] * 4:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer3(self.layer2(self.layer1(out)))
        return F.adaptive_avg_pool2d(out, 1).flatten(1)

    def forward(self, x):
        return self.fc(self.features(x))
