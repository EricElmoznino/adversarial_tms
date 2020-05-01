import torch
from utils import image_to_tensor
import os
import utils

dir = '/home/eelmozn1/datasets/adversarial_tms'
datasets = ['scenecats']
feature_names = ['conv_3']
rois = ['LOC', 'PPA', 'RANDOM']
resolution = 256

for feat_name in feature_names:
    for roi in rois:
        encoder = torch.load('saved_models/study=bold5000_featextractor=alexnet_featname={}_rois={}.pth'.format(feat_name, roi),
                            map_location=lambda storage, loc: storage)
        for dataset in datasets:
            data_dir = os.path.join(dir, dataset)
            target_dir = os.path.join(dir, 'targets_bold5000', '{}_roi={}_feat={}'.format(dataset, roi, feat_name))
            os.mkdir(target_dir)
            folder_names = os.listdir(data_dir)
            folder_names = [f for f in folder_names if f != '.DS_Store']
            for name in folder_names:
                folder = os.path.join(data_dir, name)
                image_paths = utils.listdir(folder)
                images = torch.stack([image_to_tensor(path, resolution=resolution) for path in image_paths])
                target = encoder(images).mean(dim=0)
                torch.save(target, os.path.join(target_dir, name + '.pth'))
