import torch
from torch import nn
from torchvision.models.alexnet import alexnet as alexnet_pretrained
from torchvision.models.vgg import vgg16 as vgg16_pretrained


class AlexNet(nn.Module):

    def __init__(self, feature_name):
        super().__init__()

        self.feature_name = feature_name
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
        x = self.conv_1(stimuli)
        if 'conv_1' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_2(x)
        if 'conv_2' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_3(x)
        if 'conv_3' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_4(x)
        if 'conv_4' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_5(x)
        if 'conv_5' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        if 'pool' == self.feature_name: return x
        x = self.fc_1(x)
        if 'fc_1' == self.feature_name: return x
        x = self.fc_2(x)
        if 'fc_2' == self.feature_name: return x
        x = self.fc_3(x)
        if 'fc_3' == self.feature_name: return x
        return None


class VGG16(nn.Module):

    def __init__(self, feature_name):
        super().__init__()

        self.feature_name = feature_name
        base = vgg16_pretrained(pretrained=True)

        self.conv_1 = base.features[:2]
        self.conv_2 = base.features[2:5]
        self.conv_3 = base.features[5:7]
        self.conv_4 = base.features[7:10]
        self.conv_5 = base.features[10:12]
        self.conv_6 = base.features[12:14]
        self.conv_7 = base.features[14:17]
        self.conv_8 = base.features[17:19]
        self.conv_9 = base.features[19:21]
        self.conv_10 = base.features[21:24]
        self.conv_11 = base.features[24:26]
        self.conv_12 = base.features[26:28]
        self.conv_13 = base.features[28:]
        self.avgpool = base.avgpool
        self.fc_1 = base.classifier[:3]
        self.fc_2 = base.classifier[3:6]
        self.fc_3 = base.classifier[6:]

        self.eval()

    def forward(self, stimuli):
        x = self.conv_1(stimuli)
        if 'conv_1' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_2(x)
        if 'conv_2' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_3(x)
        if 'conv_3' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_4(x)
        if 'conv_4' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_5(x)
        if 'conv_5' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_6(x)
        if 'conv_6' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_7(x)
        if 'conv_7' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_8(x)
        if 'conv_8' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_9(x)
        if 'conv_9' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_10(x)
        if 'conv_10' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_11(x)
        if 'conv_11' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_12(x)
        if 'conv_12' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.conv_13(x)
        if 'conv_13' == self.feature_name: return x.view(x.shape[0], -1)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        if 'pool' == self.feature_name: return x
        x = self.fc_1(x)
        if 'fc_1' == self.feature_name: return x
        x = self.fc_2(x)
        if 'fc_2' == self.feature_name: return x
        x = self.fc_3(x)
        if 'fc_3' == self.feature_name: return x
        return None
