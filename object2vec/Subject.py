import scipy.io


class Subject:

    def __init__(self, subject_num):
        roistack = scipy.io.loadmat('object2vec/subject_data/subj{:03}'.format(subject_num) +
                                    '/roistack.mat')['roistack']

        self.roi_names = [d[0] for d in roistack['rois'][0, 0][:, 0]]
        self.conditions = [d[0] for d in roistack['conds'][0, 0][:, 0]]

        roi_indices = roistack['indices'][0, 0][0]
        self.roi_masks = {roi: roi_indices == (i + 1) for i, roi in enumerate(self.roi_names)}

        voxels = roistack['betas'][0, 0]
        self.condition_voxels = {cond: voxels[i] for i, cond in enumerate(self.conditions)}
        self.n_voxels = voxels.shape[1]

        sets = scipy.io.loadmat('object2vec/subject_data/subj{:03}'.format(subject_num) +
                                   '/sets.mat')['sets']
        self.cv_sets = [[cond[0] for cond in s[:, 0]] for s in sets[0, :]]

    def mask_roi(self, voxels, roi):
        return voxels[self.roi_masks[roi]]
