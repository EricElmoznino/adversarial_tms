#!/usr/bin/env bash
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv1/loc --encoder_file study=object2vec_featextractor=alexnet_featname=conv_1_rois=LOC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/obj2vec_roi=LOC_feat=conv_1 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv1/ppa --encoder_file study=object2vec_featextractor=alexnet_featname=conv_1_rois=PPA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/obj2vec_roi=PPA_feat=conv_1 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv1/evc --encoder_file study=object2vec_featextractor=alexnet_featname=conv_1_rois=EVC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/obj2vec_roi=EVC_feat=conv_1 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv1/ffa --encoder_file study=object2vec_featextractor=alexnet_featname=conv_1_rois=FFA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/obj2vec_roi=FFA_feat=conv_1 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv2/loc --encoder_file study=object2vec_featextractor=alexnet_featname=conv_2_rois=LOC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/obj2vec_roi=LOC_feat=conv_2 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv2/ppa --encoder_file study=object2vec_featextractor=alexnet_featname=conv_2_rois=PPA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/obj2vec_roi=PPA_feat=conv_2 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv2/evc --encoder_file study=object2vec_featextractor=alexnet_featname=conv_2_rois=EVC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/obj2vec_roi=EVC_feat=conv_2 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv2/ffa --encoder_file study=object2vec_featextractor=alexnet_featname=conv_2_rois=FFA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/obj2vec_roi=FFA_feat=conv_2 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv3/loc --encoder_file study=object2vec_featextractor=alexnet_featname=conv_3_rois=LOC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/obj2vec_roi=LOC_feat=conv_3 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv3/ppa --encoder_file study=object2vec_featextractor=alexnet_featname=conv_3_rois=PPA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/obj2vec_roi=PPA_feat=conv_3 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv3/evc --encoder_file study=object2vec_featextractor=alexnet_featname=conv_3_rois=EVC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/obj2vec_roi=EVC_feat=conv_3 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv3/ffa --encoder_file study=object2vec_featextractor=alexnet_featname=conv_3_rois=FFA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/obj2vec_roi=FFA_feat=conv_3 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/pool/loc --encoder_file study=object2vec_featextractor=alexnet_featname=pool_rois=LOC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/obj2vec_roi=LOC_feat=pool --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/pool/ppa --encoder_file study=object2vec_featextractor=alexnet_featname=pool_rois=PPA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/obj2vec_roi=PPA_feat=pool --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/pool/evc --encoder_file study=object2vec_featextractor=alexnet_featname=pool_rois=EVC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/obj2vec_roi=EVC_feat=pool --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/pool/ffa --encoder_file study=object2vec_featextractor=alexnet_featname=pool_rois=FFA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/obj2vec_roi=FFA_feat=pool --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv1/loc --encoder_file study=object2vec_featextractor=alexnet_featname=conv_1_rois=LOC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/imagenet_roi=LOC_feat=conv_1 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv1/ppa --encoder_file study=object2vec_featextractor=alexnet_featname=conv_1_rois=PPA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/imagenet_roi=PPA_feat=conv_1 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv1/evc --encoder_file study=object2vec_featextractor=alexnet_featname=conv_1_rois=EVC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/imagenet_roi=EVC_feat=conv_1 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv1/ffa --encoder_file study=object2vec_featextractor=alexnet_featname=conv_1_rois=FFA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/imagenet_roi=FFA_feat=conv_1 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv2/loc --encoder_file study=object2vec_featextractor=alexnet_featname=conv_2_rois=LOC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/imagenet_roi=LOC_feat=conv_2 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv2/ppa --encoder_file study=object2vec_featextractor=alexnet_featname=conv_2_rois=PPA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/imagenet_roi=PPA_feat=conv_2 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv2/evc --encoder_file study=object2vec_featextractor=alexnet_featname=conv_2_rois=EVC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/imagenet_roi=EVC_feat=conv_2 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv2/ffa --encoder_file study=object2vec_featextractor=alexnet_featname=conv_2_rois=FFA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/imagenet_roi=FFA_feat=conv_2 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv3/loc --encoder_file study=object2vec_featextractor=alexnet_featname=conv_3_rois=LOC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/imagenet_roi=LOC_feat=conv_3 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv3/ppa --encoder_file study=object2vec_featextractor=alexnet_featname=conv_3_rois=PPA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/imagenet_roi=PPA_feat=conv_3 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv3/evc --encoder_file study=object2vec_featextractor=alexnet_featname=conv_3_rois=EVC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/imagenet_roi=EVC_feat=conv_3 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv3/ffa --encoder_file study=object2vec_featextractor=alexnet_featname=conv_3_rois=FFA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/imagenet_roi=FFA_feat=conv_3 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/pool/loc --encoder_file study=object2vec_featextractor=alexnet_featname=pool_rois=LOC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/imagenet_roi=LOC_feat=pool --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/pool/ppa --encoder_file study=object2vec_featextractor=alexnet_featname=pool_rois=PPA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/imagenet_roi=PPA_feat=pool --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/pool/evc --encoder_file study=object2vec_featextractor=alexnet_featname=pool_rois=EVC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/imagenet_roi=EVC_feat=pool --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/pool/ffa --encoder_file study=object2vec_featextractor=alexnet_featname=pool_rois=FFA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/imagenet_roi=FFA_feat=pool --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv1/loc --encoder_file study=object2vec_featextractor=alexnet_featname=conv_1_rois=LOC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/coco_roi=LOC_feat=conv_1 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv1/ppa --encoder_file study=object2vec_featextractor=alexnet_featname=conv_1_rois=PPA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/coco_roi=PPA_feat=conv_1 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv1/evc --encoder_file study=object2vec_featextractor=alexnet_featname=conv_1_rois=EVC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/coco_roi=EVC_feat=conv_1 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv1/ffa --encoder_file study=object2vec_featextractor=alexnet_featname=conv_1_rois=FFA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/coco_roi=FFA_feat=conv_1 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv2/loc --encoder_file study=object2vec_featextractor=alexnet_featname=conv_2_rois=LOC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/coco_roi=LOC_feat=conv_2 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv2/ppa --encoder_file study=object2vec_featextractor=alexnet_featname=conv_2_rois=PPA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/coco_roi=PPA_feat=conv_2 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv2/evc --encoder_file study=object2vec_featextractor=alexnet_featname=conv_2_rois=EVC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/coco_roi=EVC_feat=conv_2 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv2/ffa --encoder_file study=object2vec_featextractor=alexnet_featname=conv_2_rois=FFA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/coco_roi=FFA_feat=conv_2 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv3/loc --encoder_file study=object2vec_featextractor=alexnet_featname=conv_3_rois=LOC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/coco_roi=LOC_feat=conv_3 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv3/ppa --encoder_file study=object2vec_featextractor=alexnet_featname=conv_3_rois=PPA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/coco_roi=PPA_feat=conv_3 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv3/evc --encoder_file study=object2vec_featextractor=alexnet_featname=conv_3_rois=EVC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/coco_roi=EVC_feat=conv_3 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/conv3/ffa --encoder_file study=object2vec_featextractor=alexnet_featname=conv_3_rois=FFA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/coco_roi=FFA_feat=conv_3 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/pool/loc --encoder_file study=object2vec_featextractor=alexnet_featname=pool_rois=LOC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/coco_roi=LOC_feat=pool --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/pool/ppa --encoder_file study=object2vec_featextractor=alexnet_featname=pool_rois=PPA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/coco_roi=PPA_feat=pool --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/pool/evc --encoder_file study=object2vec_featextractor=alexnet_featname=pool_rois=EVC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/coco_roi=EVC_feat=pool --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/pool/ffa --encoder_file study=object2vec_featextractor=alexnet_featname=pool_rois=FFA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets/coco_roi=FFA_feat=pool --n_samples 10
