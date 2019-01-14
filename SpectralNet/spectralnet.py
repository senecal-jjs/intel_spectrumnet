import math
import torch
import torch.nn as nn
import torch.nn.init as init

class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes 
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)),
                          self.expand3x3_activation(self.expand3x3(x))], 1)


class Smash(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(Smash, self).__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size=3, padding=1, groups=inplanes, bias=False),
            nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(squeeze_planes),
            nn.ReLU(inplace=False),
        )

        self.expand = nn.Sequential(
            nn.Conv2d(squeeze_planes, squeeze_planes, kernel_size=3, padding=1, groups=squeeze_planes, bias=False),
            nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_planes),
            nn.ReLU(inplace=False),
        )

        self.squeeze1 = nn.Sequential(
            nn.Conv2d(inplanes, squeeze_planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(squeeze_planes),
            nn.ReLU(inplace=False),
        )

        self.expand1 = nn.Sequential(
            nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, padding=1),
            nn.BatchNorm2d(expand_planes),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        x = self.squeeze(x)
        return self.expand(x)


class SpectralNet(nn.Module):
    def __init__(self, version=1.0, num_classes=10, num_bands=13):
        super(SpectralNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SpectralNet version {version}" "1.0 or 1.1 expected".format(version=version))
        
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(num_bands, 96, kernel_size=2, stride=2), # modified from original architecture
                nn.ReLU(inplace=True),
                # skipped the initial maxpool due to my smaller image size
                Fire(96, 32, 64, 64),
                Fire(128, 32, 64, 64),
                Fire(128, 64, 128, 128),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), # changed kernel size
                Fire(256, 64, 128, 128),
                Fire(256, 96, 192, 192),
                Fire(384, 96, 192, 192),
                Fire(384, 128, 256, 256),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), # changed kernel size
                Fire(512, 128, 256, 256),
            )
        elif version == 1.1:
            self.features = nn.Sequential(
                nn.Conv2d(num_bands, 96, kernel_size=2, stride=2), # modified from original architecture
                nn.ReLU(inplace=False),

                # skipped the initial maxpool due to my smaller image size
                Smash(96, 16, 128),
                Smash(128, 16, 128),
                Smash(128, 32, 256),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), # changed kernel size
                Smash(256, 32, 256),
                Smash(256, 48, 384),
                Smash(384, 48, 384),
                Smash(384, 64, 512),
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), # changed kernel size 
                Smash(512, 64, 512),
            )

        # Final convolution is initialized differently from the rest 
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(8, stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)
