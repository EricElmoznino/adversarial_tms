#!/usr/bin/env bash
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/bold5000/mturkobjects/conv3/random --encoder_file study=bold5000_featextractor=alexnet_featname=conv_3_rois=RANDOM.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets_bold5000/mturksobjects_roi=RANDOM_feat=conv_3 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/bold5000/mturkobjects/conv3/loc --encoder_file study=bold5000_featextractor=alexnet_featname=conv_3_rois=LOC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets_bold5000/mturksobjects_roi=LOC_feat=conv_3 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/bold5000/mturkobjects/conv3/ppa --encoder_file study=bold5000_featextractor=alexnet_featname=conv_3_rois=PPA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets_bold5000/mturksobjects_roi=PPA_feat=conv_3 --n_samples 10
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/bold5000/mturkobjects/conv3/evc --encoder_file study=bold5000_featextractor=alexnet_featname=conv_3_rois=EVC.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets_bold5000/mturksobjects_roi=EVC_feat=conv_3 --n_samples 10
