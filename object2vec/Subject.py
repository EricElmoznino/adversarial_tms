import scipy.io


class Subject:

    def __init__(self, subject_file):
        self.subj_num = int(subject_file[-3:])
        roistack = scipy.io.loadmat(subject_file + '/roistack.mat')['roistack']

        self.roi_names = [d[0] for d in roistack['rois'][0, 0][:, 0]]

        conditions = [d[0] for d in roistack['conds'][0, 0][:, 0]]
        voxels = roistack['betas'][0, 0]
        self.condition_voxels = {cond: voxels[i] for i, cond in enumerate(conditions)}

        self.n_voxels = voxels.shape[1]

        roi_indices = roistack['indices'][0, 0][0]
        self.roi_masks = {roi: roi_indices == (i + 1) for i, roi in enumerate(self.roi_names)}
