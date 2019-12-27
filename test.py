import torch
from torchvision.models.vgg import vgg16
from utils import *
from disruption import deepdream, roi_loss_func

model = vgg16(pretrained=True)
if torch.cuda.is_available():
    model.cuda()
loss_func = roi_loss_func()

img = image_to_tensor('/home/eelmozn1/Downloads/sky1024px.jpg')
target = torch.nn.functional.one_hot(torch.tensor([1]), 1000).float().squeeze(0)

pert = deepdream(img, model, target, loss_func,
                 n_octave=6, octave_scale=1.4, alpha=0.01, n_iter=20)

tensor_to_image(pert).save('/home/eelmozn1/Downloads/output.jpg')

model.cpu()

o = model(img.unsqueeze(0))
p = model(pert.unsqueeze(0))

l_o_to_t = loss_func(o, target.unsqueeze(0)).item()
l_p_to_t = loss_func(p, target.unsqueeze(0)).item()
l_p_to_o = loss_func(o, p).item()
print('\nOriginal to Target:')
print(l_o_to_t)
print('Perturbed to Target:')
print(l_p_to_t)
print('Perturbed to Original:')
print(l_p_to_o)

c_o = o.squeeze(0).argmax().item()
c_p = p.squeeze(0).argmax().item()
print('\nOriginal class:')
print(c_o)
print('Perturbed class:')
print(c_p)
