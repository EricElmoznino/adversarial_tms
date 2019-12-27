import torch
from torchvision.models.alexnet import alexnet
from utils import *
from disruption import deepdream, roi_loss_func

model = alexnet(pretrained=True)
model = model.features
if torch.cuda.is_available():
    model.cuda()
loss_func = roi_loss_func(towards_target=False)

img = image_to_tensor('/home/eelmozn1/Downloads/sky1024px.jpg')

pert = deepdream(img, model, torch.zeros(1000), loss_func,
                 n_octave=4, octave_scale=1.4, alpha=0.01, n_iter=10)

pert = tensor_to_image(pert)
pert.save('/home/eelmozn1/Downloads/output.jpg')

