#!/usr/bin/env bash
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/bold5000/greene2009/loc_matched --encoder_file study=bold5000_featextractor=alexnet_featname=conv_3_rois=LOC_matched.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets_bold5000/greene2009_all_roi=LOC_matched_feat=conv_3 --n_samples 1
python -m gan_manipulation.optimize_for_roi --save_folder /home/eelmozn1/experiments/adversarial_tms/bold5000/greene2009/ppa_matched --encoder_file study=bold5000_featextractor=alexnet_featname=conv_3_rois=PPA_matched.pth --targets_folder /home/eelmozn1/datasets/adversarial_tms/targets_bold5000/greene2009_all_roi=PPA_matched_feat=conv_3 --n_samples 1
