from torch.nn import functional as F


def roi_loss_func(roi_mask=None, towards_target=True, loss_func=F.mse_loss):
    def roi_loss(pred, target):
        loss = loss_func(pred, target, reduction='none')
        if not towards_target and roi_mask is None:
            loss = -loss
        elif not towards_target:
            loss[:, roi_mask] = -loss[:, roi_mask]
        loss = loss.sum()
        return loss
    return roi_loss
