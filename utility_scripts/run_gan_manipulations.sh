#!/usr/bin/env bash
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/bold5000/greene2009/cca-loc --encoder_file study=bold5000_featextractor=alexnet_featname=conv_3_rois=CCA-LOC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets_bold5000/greene2009_all_roi=CCA-LOC_feat=conv_3 --n_samples 1
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/bold5000/greene2009/cca-ppa --encoder_file study=bold5000_featextractor=alexnet_featname=conv_3_rois=CCA-PPA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets_bold5000/greene2009_all_roi=CCA-PPA_feat=conv_3 --n_samples 1
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/bold5000/coco/cca-loc --encoder_file study=bold5000_featextractor=alexnet_featname=conv_3_rois=CCA-LOC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets_bold5000/coco_roi=CCA-LOC_feat=conv_3 --n_samples 1
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/bold5000/coco/cca-ppa --encoder_file study=bold5000_featextractor=alexnet_featname=conv_3_rois=CCA-PPA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets_bold5000/coco_roi=CCA-PPA_feat=conv_3 --n_samples 1
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/bold5000/bao2020/cca-loc --encoder_file study=bold5000_featextractor=alexnet_featname=conv_3_rois=CCA-LOC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets_bold5000/bao2020_roi=CCA-LOC_feat=conv_3 --n_samples 1
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/bold5000/bao2020/cca-ppa --encoder_file study=bold5000_featextractor=alexnet_featname=conv_3_rois=CCA-PPA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets_bold5000/bao2020_roi=CCA-PPA_feat=conv_3 --n_samples 1
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/bold5000/brady2008pcs_stubbiness/cca-loc --encoder_file study=bold5000_featextractor=alexnet_featname=conv_3_rois=CCA-LOC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets_bold5000/brady2008pcs_stubbiness_roi=CCA-LOC_feat=conv_3 --n_samples 1
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/bold5000/brady2008pcs_stubbiness/cca-ppa --encoder_file study=bold5000_featextractor=alexnet_featname=conv_3_rois=CCA-PPA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets_bold5000/brady2008pcs_stubbiness_roi=CCA-PPA_feat=conv_3 --n_samples 1
