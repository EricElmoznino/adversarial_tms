#!/usr/bin/env bash
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/bold5000/greene_2009/all/conv3/random --encoder_file study=bold5000_featextractor=alexnet_featname=conv_3_rois=RANDOM.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets_bold5000/greene_2009/all_roi=RANDOM_feat=conv_3 --n_samples 1
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/bold5000/greene_2009/all/conv3/ppa --encoder_file study=bold5000_featextractor=alexnet_featname=conv_3_rois=PPA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets_bold5000/greene_2009/all_roi=PPA_feat=conv_3 --n_samples 1
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/bold5000/greene_2009/all/fc1/random --encoder_file study=bold5000_featextractor=alexnet_featname=fc_1_rois=RANDOM.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets_bold5000/greene_2009/all_roi=RANDOM_feat=fc_1 --n_samples 1
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/bold5000/greene_2009/all/fc1/ppa --encoder_file study=bold5000_featextractor=alexnet_featname=fc_1_rois=PPA.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets_bold5000/greene_2009/all_roi=PPA_feat=fc_1 --n_samples 1
