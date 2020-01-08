from torch.nn import functional as F


def loss_metrics(original_stimulus, disrupted_stimulus, target, model, roi_mask=None, loss_func=F.mse_loss):
    target = target.unsqueeze(0)

    original = model(original_stimulus.unsqueeze(0))
    disrupted = model(disrupted_stimulus.unsqueeze(0))

    if roi_mask is None:
        l_o_to_t = loss_func(original, target).item()
        l_d_to_t = loss_func(disrupted, target).item()
        l_d_to_o = loss_func(disrupted, original).item()

        return {'Original to Target': l_o_to_t,
                'Disrupted to Target': l_d_to_t,
                'Disrupted to Original': l_d_to_o}

    else:
        original_on_roi, original_off_roi = original[:, roi_mask], original[:, 1 - roi_mask]
        disrupted_on_roi, disrupted_off_roi = disrupted[:, roi_mask], disrupted[:, 1 - roi_mask]
        target_on_roi, target_off_roi = target[:, roi_mask], target[:, 1 - roi_mask]

        l_o_to_t_on_roi = loss_func(original_on_roi, target_on_roi).item()
        l_d_to_t_on_roi = loss_func(disrupted_on_roi, target_on_roi).item()
        l_d_to_o_on_roi = loss_func(disrupted_on_roi, original_on_roi).item()

        l_o_to_t_off_roi = loss_func(original_off_roi, target_off_roi).item()
        l_d_to_t_off_roi = loss_func(disrupted_off_roi, target_off_roi).item()
        l_d_to_o_off_roi = loss_func(disrupted_off_roi, original_off_roi).item()

        return {'Original to Target (ON ROI)': l_o_to_t_on_roi,
                'Disrupted to Target (ON ROI)': l_d_to_t_on_roi,
                'Disrupted to Original (ON ROI)': l_d_to_o_on_roi,
                'Original to Target (OFF ROI)': l_o_to_t_off_roi,
                'Disrupted to Target (OFF ROI)': l_d_to_t_off_roi,
                'Disrupted to Original (OFF ROI)': l_d_to_o_off_roi}
