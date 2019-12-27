import torch
from torchvision.models.vgg import vgg16
from utils import *
from disruption import deepdream, roi_loss_func

model = vgg16(pretrained=True)
if torch.cuda.is_available():
    model.cuda()
layers = list(model.features.children())
model = torch.nn.Sequential(*layers[: (27 + 1)])
loss_func = roi_loss_func(towards_target=False)

img = image_to_tensor('/home/eelmozn1/Downloads/sky1024px.jpg')

pert = deepdream(img, model, torch.zeros(1000), loss_func,
                 n_octave=10, octave_scale=1.4, alpha=0.01, n_iter=20, max_jitter=0)

pert = tensor_to_image(pert)
pert.save('/home/eelmozn1/Downloads/output.jpg')

