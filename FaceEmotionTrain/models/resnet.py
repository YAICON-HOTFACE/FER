import torch
import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    def __init__(self, size=18, num_classes=1):
        super().__init__()

        if size == 18:
            self.model = models.resnet18(pretrained=True)
        elif size == 34:
            self.model = models.resnet34(pretrained=True)
        elif size == 50:
            self.model = models.resnet50(pretrained=True)
        else:
            raise NotImplementedError()

        if num_classes == 1:    # regression
            self.model.fc = nn.Sequential(
                nn.Linear(512, num_classes),
                nn.Sigmoid()
            )
        else:   # classification
            self.model.fc = nn.Sequential(
                nn.Linear(512, num_classes)
            )
        
    def forward(self, x):
        return self.model(x)

class ResNext(nn.Module):
    def __init__(self, size=50, ext=4, num_classes=1):
        super(ResNext, self).__init__()
        self.ext = ext
        
        if size==50 and ext==4:
            self.model = models.resnext50_32x4d(weights = 'IMAGENET1K_V1')
        elif size==101 and ext==8:
            self.model = models.resnext101_32x8d(weights = 'IMAGENET1K_V1')
        elif size==101 and ext==4:
            self.model = models.resnext101_64x4d(weights = 'IMAGENET1K_V1')
        else:
            raise ValueError("Not available model")

        if num_classes == 1:    # regression
            self.model.fc = nn.Sequential(
                nn.Linear(512*self.ext, num_classes),
                nn.Sigmoid()
            )
        else:   # classification
            self.model.fc = nn.Sequential(
                nn.Linear(512*self.ext, num_classes)
            )

    def forward(self, x):
        return self.model(x)