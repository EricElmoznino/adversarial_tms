import torch
from torch import nn
from utils import imagenet_mean


class DeePSiM(nn.Module):

    def __init__(self):
        super().__init__()
        self.n_inputs = 4096

        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.fc7 = nn.Linear(self.n_inputs, 4096)
        self.fc6 = nn.Linear(4096, 4096)
        self.fc5 = nn.Linear(4096, 4096)
        self.tconv5_0 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False)
        self.tconv5_1 = nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1, bias=False)
        self.tconv4_0 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False)
        self.tconv4_1 = nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1, bias=False)
        self.tconv3_0 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False)
        self.tconv3_1 = nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.tconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)
        self.tconv1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False)
        self.tconv0 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1, bias=False)

        self.eval()
        self.load_state_dict(torch.load('gan_manipulation/pretrained_models/deepsim.pth',
                                        map_location=lambda storage, loc: storage))

        self.imagenet_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)

    def forward(self, x):
        lrelu = self.lrelu
        x = lrelu(self.fc7(x))
        x = lrelu(self.fc6(x))
        x = lrelu(self.fc5(x))
        x = x.view(-1, 256, 4, 4)
        x = lrelu(self.tconv5_0(x))
        x = lrelu(self.tconv5_1(x))
        x = lrelu(self.tconv4_0(x))
        x = lrelu(self.tconv4_1(x))
        x = lrelu(self.tconv3_0(x))
        x = lrelu(self.tconv3_1(x))
        x = lrelu(self.tconv2(x))
        x = lrelu(self.tconv1(x))
        x = self.tconv0(x)
        x = self.deprocess(x)
        return x

    def deprocess(self, x):
        x = x.clone()
        x = x[:, [2, 1, 0], :, :]                                               # BGR to RGB
        x /= 255                                                                # Rescale
        x += self.imagenet_mean                                                 # Undo ImageNet normalization
        return x

    def cuda(self):
        super().cuda()
        self.imagenet_mean = self.imagenet_mean.cuda()

    def cpu(self):
        super().cpu()
        self.imagenet_mean = self.imagenet_mean.cpu()
