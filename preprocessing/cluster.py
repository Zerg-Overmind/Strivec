# dbscan clustering
import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
import torch
import numpy as np
torch.manual_seed(0)
np.random.seed(0)


from opt_adapt import config_parser
args = config_parser()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_ids
from models.apparatus import *
from utils import *
import numpy as np
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN, KMeans, MeanShift, SpectralClustering
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot
from mvs import mvs_utils, filter_utils

path = "/home/xharlie/dev/cloud_tensoRF/log/ship_adapt_0.4_0.2/clusters/"

import numpy as np

def scatter_mean(indices, updates, shape):
    target = np.zeros(shape, dtype=updates.dtype)
    divd = np.zeros(shape, dtype=updates.dtype)
    indices = tuple(indices.reshape(1, -1))
    np.add.at(target, indices, updates)
    np.add.at(divd, indices, 1)

    return target / divd



def load_pnts():
	vox_res=100
	pointfile="/home/xharlie/dev/cloud_tensoRF/log/ship_points.txt"
	xyz_world_all = torch.as_tensor(np.loadtxt(pointfile, delimiter=";"), dtype=torch.float32)
	xyz_world_all, _, sampled_pnt_idx = mvs_utils.construct_vox_points_closest(
		xyz_world_all.cuda() if len(xyz_world_all) < 99999999 else xyz_world_all[::(len(xyz_world_all) // 99999999 + 1), ...].cuda(), vox_res)
	return xyz_world_all[:,:3].cpu().numpy()

def cluster(X, method="gm", num=100, vis=False):
	# define the model
	# ‘full’: each component has its own general covariance matrix.
	# ‘tied’: all components share the same general covariance matrix.
	# ‘diag’: each component has its own diagonal covariance matrix.
	# ‘spherical’: each component has its own single variance.

	if method == "gm":
		model = GaussianMixture(n_components=num, covariance_type="full", init_params="k-means++", max_iter=1000)
	elif method == "km":
		model = KMeans(n_clusters=num, init="k-means++")
	elif method == "ms":
		model = MeanShift(n_jobs=18)
	elif method == "sc":
		model = SpectralClustering(n_clusters=num, n_jobs=18)
	else:
		model = DBSCAN(eps=0.20, min_samples=9, n_jobs=18)

	# fit model and predict clusters
	print("start clustering using ", method)
	cluster_inds = np.asarray(model.fit_predict(X))
	# score_samples = np.asarray(model.score_samples(X))
	# # scores = np.asarray(model.score(X))
	# cluster_scores = np.asarray(model.predict_proba(X))
	# print("cluster_scores", cluster_scores.shape)
	# max_cluster_scores = np.max(cluster_scores, axis=1)
	# max_cluster_scores_inds = np.argmax(cluster_scores, axis=1)
	# sum_cluster_scores = np.sum(cluster_scores, axis=1)
	# min_inds = np.argsort(max_cluster_scores)
	# # mincluster_inds = np.argmin(cluster_scores, axis=0)
	# print("cluster_scores", max_cluster_scores)
	# print("score_min", max_cluster_scores[min_inds[100]], max_cluster_scores[min_inds[1000]], max_cluster_scores[min_inds[3000]], max_cluster_scores[min_inds[10000]])
	#
	#
	# # print("max_cluster_scores_inds", max_cluster_scores_inds.shape, max_cluster_scores.shape)
	# # max_cluster_mean = scatter_mean(max_cluster_scores_inds, max_cluster_scores, [num])
	# # print("max_cluster_mean", max_cluster_mean)
	# # max_cluster_mean_inds = np.argsort(max_cluster_mean)
	# # print("smallest_cluster_inds", max_cluster_mean_inds)
	#
	# prior = np.asarray(model.weights_)
	# p_v = prior / np.linalg.norm(np.asarray(model.covariances_), axis=(1, 2))
	# print("prior weight", prior)
	# print("prior weight min sort indices ", np.argsort(prior))
	# print("**prior weight over norm min sort indices ", np.argsort(p_v))
	# print("**prior weight over norm", p_v[np.argsort(p_v)])

	# os.makedirs(path, exist_ok=True)
	# np.savetxt(os.path.join(path, "mininds_100.txt"), X[min_inds[:100]], delimiter=";")
	# np.savetxt(os.path.join(path, "mininds_1000.txt"), X[min_inds[:1000]], delimiter=";")
	# np.savetxt(os.path.join(path, "mininds_3000.txt"), X[min_inds[:3000]], delimiter=";")
	# np.savetxt(os.path.join(path, "mininds_10000.txt"), X[min_inds[:10000]], delimiter=";")
	# print(cluster_inds)
	# retrieve unique clusters
	clusters = np.unique(cluster_inds)
	print("finished with ", len(clusters), " clusters", clusters)
	cluster_xyz=np.zeros([num,3], dtype=np.float32)
	cluster_mask=np.ones([num], dtype=bool)
	# create scatter plot for samples from each cluster
	if vis:
		counter = 0
		for i in range(len(clusters)):
			row_mask = cluster_inds == clusters[i]
			# print("row_ix", row_mask.shape)
			os.makedirs(path, exist_ok=True)
			np.savetxt(os.path.join(path, "cluster_{:04d}.txt".format(counter)), X[row_mask], delimiter=";")
			cluster_xyz[i] = np.mean(X[row_mask], axis=0)
			counter+=1
	return cluster_xyz, cluster_mask, cluster_inds

if __name__ == '__main__':
	X = load_pnts()
	np.savetxt("/home/xharlie/dev/cloud_tensoRF/log/ship_adapt_0.4_0.2_try/clusters/ship.txt", X, delimiter=";")
	print("X.shape", X.shape)
	cluster(X, vis=True)
