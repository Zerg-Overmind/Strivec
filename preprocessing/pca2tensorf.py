import numpy as np
from sklearn.decomposition import PCA
import math

def mask_split(pnts, indices, mask):
    unique = np.sort(np.unique(indices))
    print("unique", unique)
    times = len(indices) // len(pnts)
    rpnts = pnts.repeat(times, 1)
    if mask is None:
        return [rpnts[indices == i] for i in unique], None
    return [rpnts[indices == i] for i in unique if mask[i]], [rpnts[indices == i] for i in unique if not mask[i]]


def from_rot_2_Euler(R):
    R = np.array(R)
    # sanity check when R=I
    #R = np.eye(R.shape[0])
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
       x = math.atan2(R[2,1] , R[2,2])
       y = math.atan2(-R[2,0], sy)
       z = math.atan2(R[1,0], R[0,0])
    else:
       x = math.atan2(-R[1,2], R[1,1])
       y = math.atan2(-R[2,0], sy)
       z = 0
    return [x, y, z]


def init_pca_svd(cluster_xyz, cluster_mask, pnts, cluster_inds):
    pca_cluster_newpnts, pca_axis, R_axis = [], [], []
    cluster_raw_mean = np.zeros([len(cluster_xyz), 3], dtype=np.float32)
    pca_cluster_edge_leng = np.zeros([len(cluster_xyz), 3], dtype=np.float32)
    cluster_raw_center = np.zeros([len(cluster_xyz), 3], dtype=np.float32)
    stds = np.zeros([len(cluster_xyz), 3], dtype=np.float32)
    # print("pnts", pnts.shape, inv_idx_lst.shape)
    cluster_pnts, cluster_pnts_leftout = mask_split(pnts, cluster_inds, cluster_mask)
    for k_c in range(len(cluster_pnts)):
        pnts_data = cluster_pnts[k_c].numpy()
        if len(pnts_data) > 3:
            pca_k = PCA(n_components=3)
            new_pnts = pca_k.fit_transform(pnts_data)  # transformed pnts in new coords
            cluster_raw_mean[k_c] = pca_k.mean_
            pca_axis.append(pca_k.components_.astype(np.float32))
            # stds[k_c].append((S+S.max())/S.max())
            # stds[k_c] = np.maximum(np.max(pca_k.singular_values_) / 5, pca_k.singular_values_)
            stds[k_c] = pca_k.singular_values_ / np.sqrt(len(pnts_data))
        else:
            cluster_raw_mean[k_c] = np.mean(pnts_data, axis=0)
            new_pnts = pnts_data - cluster_raw_mean[k_c][None, :]
            pca_axis.append(np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32))
            stds[k_c] = np.std(new_pnts, axis=0)
        pca_min, pca_max = np.min(new_pnts, axis=0), np.max(new_pnts, axis=0)
        pca_cluster_newpnts.append(new_pnts - (pca_min + pca_max) / 2)
        pca_cluster_edge_leng[k_c, :] = (pca_max - pca_min)
        R_axis.append(from_rot_2_Euler(pca_axis[k_c].T))
        cluster_raw_center[k_c, :] = np.matmul((pca_min + pca_max) / 2, pca_axis[k_c]).T + cluster_raw_mean[k_c]

    #  geo_cluster: raw points in each cluster; pca_cluster_newpnts: new basis aligned pnts, pca_cluster_edge_leng: edge length of pca_cluster_newpnts, cluster_raw_mean: mean center of raw_cluster, cluster_raw_center: geo center of raw cluster, pca_axis: 3 new basis, R_axis: 3 euler angles, stds: adjusted singluar values for 3 basis

    return cluster_pnts, cluster_pnts_leftout, pca_cluster_newpnts, pca_cluster_edge_leng, cluster_raw_mean, cluster_raw_center, pca_axis, R_axis, stds

#
# def set_local_range(geo_xyz, pca_cluster_edge_leng, cluster_pnts, pca_cluster_newpnts, pca_axis, stds):
#     local_dims, local_range = [], []
#     for l in range(self.lvl):
#         unit = self.args.local_unit[l][self.up_stage]
#         print("unit of level {} is {}".format(self.lvl, unit))
#         print("stds[l]", stds[l].shape)
#         dims = np.ceil((np.minimum(pca_cluster_edge_leng[l], stds[l] * 5) * self.args.dilation_ratio[l] - unit) * 0.5  / unit).astype(np.int16)
#         local_dims.append((torch.as_tensor(dims, dtype=torch.int16, device=self.device) * 2 + 1).contiguous())
#         local_range.append(torch.as_tensor((dims+0.5) * unit, dtype=torch.float32, device=self.device).contiguous())
#     return local_range, local_dims


def find_tensorf_box(cluster_xyz, cluster_mask, pnts, cluster_inds):
    cluster_pnts, cluster_pnts_leftout, pca_cluster_newpnts, pca_cluster_edge_leng, cluster_raw_mean, cluster_raw_center, pca_axis, pnt_rot, stds = init_pca_svd(cluster_xyz, cluster_mask, pnts, cluster_inds)
    cluster_xyz, box_length = set_box(cluster_xyz, cluster_raw_center, pca_cluster_edge_leng, cluster_pnts, pca_cluster_newpnts, pca_axis, stds)
    outpnts = pnts_uncovered(cluster_pnts, pca_cluster_newpnts, box_length)
    outpnts = cluster_pnts_leftout + outpnts if cluster_pnts_leftout is not None else outpnts
    if len(outpnts) >= 2:
        outpnts = np.concatenate(outpnts)
    elif len(outpnts) == 1:
        outpnts = outpnts[0]
    else:
        outpnts = None
    np.savetxt("/home/xharlie/dev/cloud_tensoRF/log/ship_adapt_0.4_0.2/rot_tensoRF/leftout_pnt.txt", outpnts, delimiter=";")
    return cluster_pnts, cluster_xyz, box_length, pca_axis, stds, outpnts

def set_box(geo_xyz, cluster_raw_center, pca_cluster_edge_leng, cluster_pnts, pca_cluster_newpnts, pca_axis, stds):
    box_length = np.minimum(pca_cluster_edge_leng, stds * 5)
    return cluster_raw_center, box_length

def pnts_uncovered(cluster_pnts, pca_cluster_newpnts, box_length):
    outpnts = []
    for pnt, newpnt, box in zip(cluster_pnts, pca_cluster_newpnts, box_length):
        mask = np.any(np.abs(newpnt) > (box / 2), axis=-1)
        if mask.any():
            outpnts.append(pnt[mask])
    return outpnts