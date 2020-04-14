import torch
from utils import image_to_tensor
import os

dir = '/home/eelmozn1/datasets/adversarial_tms'
datasets = ['imagenet', 'coco', 'obj2vec']
feature_names = ['conv_1', 'conv_2', 'conv_3', 'pool']
rois = ['EVC', 'PPA', 'LOC', 'FFA']
resolution = 256

for feat_name in feature_names:
    for roi in rois:
        encoder = torch.load('saved_models/study=object2vec_featextractor=alexnet_featname={}_rois={}.pth'.format(feat_name, roi),
                            map_location=lambda storage, loc: storage)
        for dataset in datasets:
            data_dir = os.path.join(dir, dataset)
            target_dir = os.path.join(dir, 'targets', '{}_roi={}_feat={}'.format(dataset, roi, feat_name))
            os.mkdir(target_dir)
            image_names = os.listdir(data_dir)
            images_names = [img for img in image_names if img != '.DS_Store']
            for name in images_names:
                path = os.path.join(data_dir, name)
                image = image_to_tensor(path, resolution=resolution)
                target = encoder(image.unsqueeze(dim=0)).squeeze(dim=0)
                torch.save(target, os.path.join(target_dir, name + '.pth'))
