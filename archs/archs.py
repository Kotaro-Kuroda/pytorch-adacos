import numpy as np

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision
from metrics import AdaCos

__all__ = ['ResNet', 'GhostNet', 'GhostNet1D', 'MNISTNet']


class ResNet(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()

        if args.backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            last_channels = 512
        elif args.backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            last_channels = 512
        elif args.backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            last_channels = 2048
        elif args.backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=True)
            last_channels = 2048
        elif args.backbone == 'resnet152':
            self.backbone = models.resnet152(pretrained=True)
            last_channels = 2048
        self.features = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4)

        self.bn1 = nn.BatchNorm2d(last_channels)
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear(8 * 8 * last_channels, args.num_features)
        self.bn2 = nn.BatchNorm1d(args.num_features)
        self.adacos = AdaCos(args.num_features, num_classes)

    def freeze_bn(self):
        for m in self.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x, label=None):
        x = self.features(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        output = self.bn2(x)
        output = self.adacos(output, label)
        return output


class GhostNet(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        backbone = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=True)
        last_channels = 960
        self.features = nn.Sequential(
            backbone.conv_stem,
            backbone.bn1,
            backbone.act1,
            backbone.blocks
        )
        self.bn1 = nn.BatchNorm2d(last_channels)
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear(16 * 16 * last_channels, args.num_features)
        self.bn2 = nn.BatchNorm1d(args.num_features)
        self.adacos = AdaCos(args.num_features, num_classes)

    def freeze_bn(self):
        for m in self.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x, label=None):
        x = self.features(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.bn2(x)
        output = self.adacos(x, label)
        return output


class GhostNet1D(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        backbone = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=False)
        last_channels = 960
        backbone.conv_stem = torch.nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.features = nn.Sequential(
            backbone.conv_stem,
            backbone.bn1,
            backbone.act1,
            backbone.blocks
        )
        self.bn1 = nn.BatchNorm2d(last_channels)
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear(16 * 16 * last_channels, args.num_features)
        self.bn2 = nn.BatchNorm1d(args.num_features)
        self.adacos = AdaCos(args.num_features, num_classes)

    def freeze_bn(self):
        for m in self.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def forward(self, x, label=None):
        x = self.features(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.bn2(x)
        output = self.adacos(x, label)
        return output


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.relu(x)

        return output


class MNISTNet(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()

        self.pool = nn.MaxPool2d((2, 2))
        self.features = nn.Sequential(*[
            VGGBlock(1, 16, 16),
            self.pool,
            VGGBlock(16, 32, 32),
            self.pool,
            VGGBlock(32, 64, 64),
            self.pool,
        ])

        self.bn1 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout2d(0.5)
        self.fc = nn.Linear(3 * 3 * 64, args.num_features)
        self.bn2 = nn.BatchNorm1d(args.num_features)
        self.adacos = AdaCos(args.num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input, label=None):
        x = self.features(input)
        x = self.bn1(x)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        output = self.bn2(x)
        output = self.adacos(output, label)
        return output
