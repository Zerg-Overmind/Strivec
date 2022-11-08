import numpy as np
from sklearn.decomposition import PCA
import math, sys, os, pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.parent.absolute(), '..'))
from models.apparatus import draw_box_pca
import torch
from itertools import compress
from compas.geometry import oriented_bounding_box_numpy


def mask_split(pnts, indices):
    unique = np.sort(np.unique(indices))
    print("unique mask_split ", unique)
    times = len(indices) // len(pnts)
    rpnts = pnts.repeat(times, 1)
    return [rpnts[indices == i] for i in unique]

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


def cluster_svd(cluster_xyz, cluster_pnts, cluster_model):
    # print("cluster_model.covariances_", np.asarray(cluster_model.covariances_).shape)
    pca_cluster_newpnts, R_axis = [], []
    pca_axis = np.zeros([len(cluster_xyz), 3, 3], dtype=np.float32)
    pca_cluster_edge_leng = np.zeros([len(cluster_xyz), 3], dtype=np.float32)
    cluster_raw_center = np.zeros([len(cluster_xyz), 3], dtype=np.float32)
    stds = np.zeros([len(cluster_xyz), 3], dtype=np.float32)
    cluster_raw_mean = np.asarray(cluster_model.means_)[...,:3] #np.zeros([len(cluster_xyz), 3], dtype=np.float32)
    eigen_vals, eigen_vecs = np.linalg.eig(np.asarray(cluster_model.covariances_))
    idx = np.argsort(eigen_vals, axis=1)[..., ::-1]
    for k_c in range(len(eigen_vals)):
        stds[k_c] = np.sqrt(eigen_vals[k_c][idx[k_c, :3]])
        pca_axis[k_c] = eigen_vecs[k_c][idx[k_c, :3], idx[k_c, :3]].T
        # print("diff cluster_raw_mean",cluster_raw_mean[k_c] - np.mean(pnts_data[..., :3], axis=0))
        cluster_raw_center[k_c] = cluster_raw_mean[k_c]
        new_pnts = (cluster_pnts[k_c][:, :3] - cluster_raw_mean[k_c][None, :])
        new_pnts = np.matmul(new_pnts, pca_axis[k_c].T)
        pca_cluster_newpnts.append(new_pnts)
        pca_cluster_edge_leng[k_c, :] = np.minimum(np.abs(np.min(new_pnts, axis=0)), np.max(new_pnts, axis=0)) * 2
        R_axis.append(from_rot_2_Euler(pca_axis[k_c].T))
    # print("np.sqrt(eigen_vals)", np.sqrt(eigen_vals), eigen_vals.shape)
    return cluster_pnts, pca_cluster_newpnts, pca_cluster_edge_leng, cluster_raw_mean, cluster_raw_center, pca_axis, R_axis, stds


def pca_svd_center(cluster_xyz, cluster_pnts, cluster_model):
    pca_cluster_newpnts, R_axis = [], []
    cluster_raw_mean = np.zeros([len(cluster_xyz), 3], dtype=np.float32)
    pca_axis = np.zeros([len(cluster_xyz), 3, 3], dtype=np.float32)
    pca_cluster_edge_leng = np.zeros([len(cluster_xyz), 3], dtype=np.float32)
    cluster_raw_center = np.zeros([len(cluster_xyz), 3], dtype=np.float32)
    stds = np.zeros([len(cluster_xyz), 3], dtype=np.float32)
    for k_c in range(len(cluster_pnts)):
        pnts_data = cluster_pnts[k_c]
        if len(pnts_data) > 3:
            pca_k = PCA(n_components=3, svd_solver='full')
            new_pnts = pca_k.fit_transform(pnts_data[..., :3])  # transformed pnts in new coords
            cluster_raw_mean[k_c] = pca_k.mean_
            pca_axis[k_c] = pca_k.components_.astype(np.float32)
            # stds[k_c].append((S+S.max())/S.max())
            # stds[k_c] = np.maximum(np.max(pca_k.singular_values_) / 5, pca_k.singular_values_)
            stds[k_c] = pca_k.singular_values_ / np.sqrt(len(pnts_data))
        else:
            cluster_raw_mean[k_c] = np.mean(pnts_data[..., :3], axis=0)
            new_pnts = pnts_data[..., :3] - cluster_raw_mean[k_c][None, :]
            pca_axis[k_c] = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
            stds[k_c] = np.std(new_pnts, axis=0)
        pca_min, pca_max = np.min(new_pnts, axis=0), np.max(new_pnts, axis=0)
        pca_cluster_newpnts.append(new_pnts - (pca_min + pca_max) / 2)
        pca_cluster_edge_leng[k_c, :] = (pca_max - pca_min)
        R_axis.append(from_rot_2_Euler(pca_axis[k_c].T))
        cluster_raw_center[k_c, :] = np.matmul((pca_min + pca_max) / 2, pca_axis[k_c]).T + cluster_raw_mean[k_c]
    return cluster_pnts, pca_cluster_newpnts, pca_cluster_edge_leng, cluster_raw_mean, cluster_raw_center, pca_axis, R_axis, stds


def pca_svd_mean(cluster_xyz, cluster_pnts, cluster_model):
    pca_cluster_newpnts, R_axis = [], []
    cluster_raw_mean = np.zeros([len(cluster_xyz), 3], dtype=np.float32)
    pca_axis = np.zeros([len(cluster_xyz), 3, 3], dtype=np.float32)
    pca_cluster_edge_leng = np.zeros([len(cluster_xyz), 3], dtype=np.float32)
    cluster_raw_center = np.zeros([len(cluster_xyz), 3], dtype=np.float32)
    stds = np.zeros([len(cluster_xyz), 3], dtype=np.float32)
    for k_c in range(len(cluster_pnts)):
        pnts_data = cluster_pnts[k_c]
        if len(pnts_data) > 3:
            pca_k = PCA(n_components=3, svd_solver='full')
            new_pnts = pca_k.fit_transform(pnts_data[..., :3])  # transformed pnts in new coords
            cluster_raw_mean[k_c] = pca_k.mean_
            pca_axis[k_c] = pca_k.components_.astype(np.float32)
            # stds[k_c].append((S+S.max())/S.max())
            # stds[k_c] = np.maximum(np.max(pca_k.singular_values_) / 5, pca_k.singular_values_)
            stds[k_c] = pca_k.singular_values_ / np.sqrt(len(pnts_data))
        else:
            cluster_raw_mean[k_c] = np.mean(pnts_data[..., :3], axis=0)
            new_pnts = pnts_data[..., :3] - cluster_raw_mean[k_c][None, :]
            pca_axis[k_c] = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
            stds[k_c] = np.std(new_pnts, axis=0)
        pca_cluster_newpnts.append(new_pnts)
        pca_cluster_edge_leng[k_c, :] = np.minimum(np.abs(np.min(new_pnts, axis=0)), np.max(new_pnts, axis=0)) * 2
        # pca_cluster_edge_leng[k_c, :] = np.maximum(np.abs(np.min(new_pnts, axis=0)), np.max(new_pnts, axis=0)) * 2
        R_axis.append(from_rot_2_Euler(pca_axis[k_c].T))
        cluster_raw_center[k_c, :] = cluster_raw_mean[k_c]
    return cluster_pnts, pca_cluster_newpnts, pca_cluster_edge_leng, cluster_raw_mean, cluster_raw_center, pca_axis, R_axis, stds


def get_obb(cluster_xyz, cluster_pnts, cluster_model):
    pca_cluster_newpnts, R_axis = [], []
    pca_axis = np.zeros([len(cluster_xyz), 3, 3], dtype=np.float32)
    pca_cluster_edge_leng = np.zeros([len(cluster_xyz), 3], dtype=np.float32)
    cluster_raw_center = np.zeros([len(cluster_xyz), 3], dtype=np.float32)
    stds = np.zeros([len(cluster_xyz), 3], dtype=np.float32)
    cluster_raw_mean = np.asarray(cluster_model.means_)[..., :3]  # np.zeros([len(cluster_xyz), 3],
    eigen_vals, eigen_vecs = np.linalg.eig(np.asarray(cluster_model.covariances_))
    idx = np.argsort(eigen_vals, axis=1)[..., ::-1]
    for k_c in range(len(cluster_xyz)):
        pnts_data = cluster_pnts[k_c][:, :3]
        if len(pnts_data) > 4:
            box = oriented_bounding_box_numpy(pnts_data)
            bbox = np.stack([box[4,:] - box[0,:], box[1,:] - box[0,:], box[3,:] - box[0,:]], axis=0)
            bbox_leng = np.linalg.norm(bbox, axis=1)
            bbox_axis_order = np.argsort(bbox_leng)[::-1]
            # print("bbox_leng", bbox_leng)
            pca_axis[k_c] = (bbox[bbox_axis_order] / bbox_leng[bbox_axis_order, None])
            cluster_raw_center[k_c] = np.mean(box, axis=0)
            new_pnts = (pnts_data - cluster_raw_center[k_c][None, :])
            new_pnts = np.matmul(new_pnts, pca_axis[k_c].T)
            stds[k_c] = np.std(new_pnts, axis=0)
        else:
            cluster_raw_mean[k_c] = np.mean(pnts_data[..., :3], axis=0)
            cluster_raw_center[k_c] = cluster_raw_mean[k_c]
            new_pnts = pnts_data[..., :3] - cluster_raw_mean[k_c][None, :]
            pca_axis[k_c] = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
            stds[k_c] = np.std(new_pnts, axis=0)
        pca_cluster_newpnts.append(new_pnts)
        pca_cluster_edge_leng[k_c, :] = np.minimum(np.abs(np.min(new_pnts, axis=0)), np.max(new_pnts, axis=0)) * 2
        R_axis.append(from_rot_2_Euler(pca_axis[k_c].T))
    return cluster_pnts, pca_cluster_newpnts, pca_cluster_edge_leng, cluster_raw_mean, cluster_raw_center, pca_axis, R_axis, stds


def find_tensorf_box(cluster_xyz, pnts, cluster_inds, cluster_model, boxing_method):
    box_func = pca_svd_mean
    if boxing_method == "pca_center":
        box_func = pca_svd_center
    elif boxing_method == "obb":
        box_func = get_obb
    elif boxing_method == "gmm":
        box_func = cluster_svd
    cluster_pnts = mask_split(pnts, cluster_inds)

    cluster_pnts, pca_cluster_newpnts, pca_cluster_edge_leng, cluster_raw_mean, cluster_raw_center, pca_axis, pnt_rot, stds = box_func(cluster_xyz, cluster_pnts, cluster_model) #pca_svd_center
    # print("stds", stds)
    # exit()
    cluster_xyz, box_length = set_box(cluster_raw_center, cluster_raw_center, pca_cluster_edge_leng, cluster_pnts, pca_cluster_newpnts, pca_axis, stds)

    return cluster_pnts, pca_cluster_newpnts, cluster_xyz, box_length, pca_axis, stds


def set_box(geo_xyz, cluster_raw_center, pca_cluster_edge_leng, cluster_pnts, pca_cluster_newpnts, pca_axis, stds):
    # box_length = np.maximum(pca_cluster_edge_leng, 0.000001)
    box_length = np.maximum(np.minimum(pca_cluster_edge_leng, stds * 5), 0.000001)
    # box_length = np.maximum(np.minimum(pca_cluster_edge_leng, stds * 5), 0.000001)
    # box_length = np.maximum(pca_cluster_edge_leng, 0.00001)
    return geo_xyz, box_length

def pnts_uncovered(cluster_pnts, pca_cluster_newpnts, box_length, dilation):
    outpnts = []
    dilation = 1
    for pnt, newpnt, box in zip(cluster_pnts, pca_cluster_newpnts, box_length):
        mask = np.any(np.abs(newpnt) > (box * (dilation) / 2), axis=-1)
        if mask.any():
            outpnts.append(pnt[mask])

    return outpnts

def pnts_uncovered_cross(outpnts, cluster_xyz, box_length, pca_axis, dilation):
    outpnts_mask = np.zeros([len(outpnts)], dtype=int)
    dilation = 1
    for xyz, box, axis in zip(cluster_xyz, box_length, pca_axis):

        newpnt = np.matmul(outpnts[...,:3] - xyz[None, :], axis.T)

        outpnts_mask += np.any(np.abs(newpnt) > (box * (dilation) / 2), axis=-1)

    return outpnts[outpnts_mask == len(cluster_xyz)]

def filter_cluster_n_pnts(cluster_xyz, cluster_pnts, pca_cluster_newpnts, box_length, pca_axis, stds, cluster_model, dilation, args, count=0):
    # prior = np.asarray(cluster_model.weights_)
    prior = np.asarray([len(cluster_pnt) for cluster_pnt in cluster_pnts])
    print(prior)
    # p_v = prior / np.linalg.norm(np.asarray(model.covariances_), axis=(1, 2))
    p_v = prior / np.prod(box_length, axis=-1)
    print(np.sort(p_v), np.argsort(p_v))
    draw_box_pca(torch.as_tensor(cluster_xyz), None, box_length / 2, f'{args.basedir}/{args.expname}', count+100, None, rot_m=torch.transpose(torch.as_tensor(pca_axis, dtype=torch.float32), 1, 2), subdir="rot_tensoRF")

    cluster_mask = p_v >= 50000  # 0.02
    # print(cluster_xyz.shape, len(cluster_pnts), len(pca_cluster_newpnts), box_length, pca_axis.shape, stds.shape)
    # print(cluster_mask.shape, cluster_mask)
    mask_lst = cluster_mask.tolist()
    cluster_xyz, cluster_pnts_in, cluster_pnts_out, pca_cluster_newpnts, box_length, pca_axis, stds = cluster_xyz[cluster_mask], list(compress(cluster_pnts, mask_lst)), list(compress(cluster_pnts, np.invert(cluster_mask).tolist())), list(compress(pca_cluster_newpnts, mask_lst)), box_length[cluster_mask], pca_axis[cluster_mask], stds[cluster_mask]
    if len(cluster_xyz) > 0:
        draw_box_pca(torch.as_tensor(cluster_xyz), None, box_length/2, f'{args.basedir}/{args.expname}', count, None, rot_m=torch.transpose(torch.as_tensor(pca_axis, dtype=torch.float32), 1, 2), subdir="rot_tensoRF")
    box_length = np.maximum(box_length, 0.07)

    outpnts = pnts_uncovered(cluster_pnts_in, pca_cluster_newpnts, box_length, dilation)

    outpnts = outpnts + cluster_pnts_out
    if len(outpnts) >= 2:
        outpnts = np.concatenate(outpnts)
    elif len(outpnts) == 1:
        outpnts = outpnts[0]
    else:
        outpnts = None
    if outpnts is not None and len(outpnts) > 0:
        outpnts = pnts_uncovered_cross(outpnts, cluster_xyz, box_length, pca_axis, dilation)

    if outpnts is not None and len(outpnts) > 0:
        np.savetxt(f'{args.basedir}/{args.expname}'+"/rot_tensoRF/output_{}.txt".format(count), outpnts, delimiter=";")
    return cluster_xyz, cluster_pnts_in, box_length, pca_axis, stds, outpnts
