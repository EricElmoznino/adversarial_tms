import torch
from torchvision.models.vgg import vgg16
from utils import *
from disruption import deepdream, roi_loss_func

model = vgg16(pretrained=True)
if torch.cuda.is_available():
    model.cuda()
loss_func = roi_loss_func()

img = image_to_tensor('/home/eelmozn1/Downloads/sky1024px.jpg')

pert = deepdream(img, model, torch.nn.functional.one_hot(torch.tensor([245]), 1000).float().squeeze(0), loss_func,
                 n_octave=6, octave_scale=1.4, alpha=0.01, n_iter=20)

pert = tensor_to_image(pert)
pert.save('/home/eelmozn1/Downloads/output.jpg')
