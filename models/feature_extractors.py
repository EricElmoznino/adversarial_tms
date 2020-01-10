import torch
from torch import nn
from torchvision.models.alexnet import alexnet as alexnet_pretrained
from torchvision.models.vgg import vgg16 as vgg16_pretrained


class AlexNet(nn.Module):

    def __init__(self, feature_names):
        super().__init__()

        self.feature_names = feature_names
        base = alexnet_pretrained(pretrained=True)

        self.conv_1 = base.features[:3]
        self.conv_2 = base.features[3:6]
        self.conv_3 = base.features[6:8]
        self.conv_4 = base.features[8:10]
        self.conv_5 = base.features[10:]
        self.avgpool = base.avgpool
        self.fc_1 = base.classifier[:3]
        self.fc_2 = base.classifier[3:6]
        self.fc_3 = base.classifier[6:]

        self.eval()

    def forward(self, stimuli):
        feats = []

        x = self.conv_1(stimuli)
        if 'conv_1' in self.feature_names: feats.append(x)
        x = self.conv_2(x)
        if 'conv_2' in self.feature_names: feats.append(x)
        x = self.conv_3(x)
        if 'conv_3' in self.feature_names: feats.append(x)
        x = self.conv_4(x)
        if 'conv_4' in self.feature_names: feats.append(x)
        x = self.conv_5(x)
        if 'conv_5' in self.feature_names: feats.append(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        if 'pool' in self.feature_names: feats.append(x)
        x = self.fc_1(x)
        if 'fc_1' in self.feature_names: feats.append(x)
        x = self.fc_2(x)
        if 'fc_2' in self.feature_names: feats.append(x)
        x = self.fc_3(x)
        if 'fc_3' in self.feature_names: feats.append(x)

        assert len(feats) == len(self.feature_names)
        feats = [f.view(f.shape[0], -1) for f in feats]
        feats = torch.cat(feats, dim=1)

        return feats


class VGG16(nn.Module):

    def __init__(self, feature_names):
        super().__init__()

        self.feature_names = feature_names
        self.base = vgg16_pretrained(pretrained=True)

        self.eval()

    def forward(self, stimuli):
        feats = []

        x = self.base.features(stimuli)
        x = self.base.avgpool(x)
        x = x.view(x.shape[0], -1)
        if 'pool' in self.feature_names: feats.append(x)
        x = self.base.classifier(x)
        if 'fc' in self.feature_names: feats.append(x)

        assert len(feats) == len(self.feature_names)
        feats = [f.view(f.shape[0], -1) for f in feats]
        feats = torch.cat(feats, dim=1)

        return feats
