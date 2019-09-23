import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-b', type=int, default=0)
parser.add_argument('-e', type=int, default=-1)
parser.add_argument('--epoch', type=int, required=True)
args = parser.parse_args()

import os

import numpy as np
import skimage.measure
import csv
from glob import glob

import tqdm

# Parameters
SNAPSHOT_EPOCH = args.epoch

DATA_FILE_FILTER = './data/stage1/*'
DATA_DIR = './data/preprocessed_masks/lung_region/'
PROB_DIR = './data/preprocessed_masks/nodule_prob_with_lidc_epoch_%08d/' % SNAPSHOT_EPOCH
SAVE_DIR = './data/preprocessed_masks/cluster_candidates_with_lidc/'

MAX_SIZE = 4/3*np.pi*(20**3)
PROB_THRESHOLD = 0.75
THRESHOLD_STEP = 0.05

TOP_NODULE_NUM = 20
CROP_SIZE = 32

# Load list
file_paths = glob(DATA_FILE_FILTER)
print('number of files:', len(file_paths))
file_paths = file_paths[args.b:args.e]
print('number of files in range %d - %d: %d' % (args.b, args.e, len(file_paths)))

os.makedirs(SAVE_DIR, exist_ok=True)
for n in tqdm.trange(len(file_paths)):
    path = file_paths[n]
    uid = os.path.basename(path)
    
    # Step 1: load data
    # Load the image and the lung mask
    f = np.load(os.path.join(DATA_DIR, uid + '_lung_region.npz'))
    lung_img, lung_mask = f['lung_img'], f['lung_mask']
    lung_img[~lung_mask] = -1024
    f.close()
    # Load the 3D probability map
    f = np.load(os.path.join(PROB_DIR, uid + '_nodule_prob.npz'))
    prob3d_all = f['prob3d']
    prob3d = prob3d_all[:, :, :, 0]
    f.close()
    Z, Y, X = lung_img.shape
    # print('CT scan size in ZYX:', lung_img.shape)
    
    # Step 2: get nodule clusters
    nodule_mask = (prob3d >= PROB_THRESHOLD)
    # Find connected regions
    # Get the cluster id (integer) of each voxel, and number of clusters
    cluster_ids, cluster_num = skimage.measure.label(nodule_mask, return_num=True, background=0)
    # the cluster num above is equal to the maximum in cluster_ids
    # so add 1 get the "actual" cluster number (including the bg)
    cluster_num += 1
    # print('number of clusters found:', cluster_num)
    
    # Step 3: refine clusters
    # skipped
    
    # Step 4: take the top N cluster, sorted by probability weights
    # compute the sum of nodule probability within the each cluster
    prob_sums = np.array([np.sum(prob3d[cluster_ids == i]) for i in range(cluster_num)])
    prob_sums[0] = 0  # skip 0, the background
    topN_cluster_inds = np.argsort(prob_sums)[::-1][:TOP_NODULE_NUM]

    # # See how much mess the top-N encodes
    # prob_sum_topN = np.sum(prob_sums[topN_cluster_inds])
    # prob_sum_thresh = np.sum(prob3d[prob3d >= PROB_THRESHOLD])
    # print('fraction of probs in the top %d: %f' % (TOP_NODULE_NUM, prob_sum_topN/prob_sum_thresh))

    # compute the centroid of each cluster in the top-N and take crops
    half_size = CROP_SIZE // 2
    topN_centroids = np.zeros((TOP_NODULE_NUM, 3))
    topN_lung_img_crop = np.zeros((TOP_NODULE_NUM, CROP_SIZE, CROP_SIZE, CROP_SIZE))
    topN_prob3d_all_crop = np.zeros((TOP_NODULE_NUM, CROP_SIZE, CROP_SIZE, CROP_SIZE, 2))
    Z_mesh, Y_mesh, X_mesh = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing='ij')
    # # check whether the meshgrid is currenctly computed
    # z, y, x = 20, 30, 40
    # assert(np.all(Z_mesh[z, :, :] == z))
    # assert(np.all(Y_mesh[:, y, :] == y))
    # assert(np.all(X_mesh[:, :, x] == x))
    for n_cluster, i in enumerate(topN_cluster_inds):
        cluster_prob3d = prob3d * (cluster_ids == i)
        # normalize to have sum equal to 1
        cluster_prob3d = cluster_prob3d / np.sum(cluster_prob3d)
        z = int(np.sum(Z_mesh*cluster_prob3d))
        y = int(np.sum(Y_mesh*cluster_prob3d))
        x = int(np.sum(X_mesh*cluster_prob3d))
        z = np.minimum(np.maximum(z, half_size), Z-half_size)
        y = np.minimum(np.maximum(y, half_size), Y-half_size)
        x = np.minimum(np.maximum(x, half_size), X-half_size)
        topN_centroids[n_cluster] = [z, y, x]

        # Take crop from the centroid
        z_begin, z_end = z - half_size, z + half_size
        y_begin, y_end = y - half_size, y + half_size
        x_begin, x_end = x - half_size, x + half_size
        topN_lung_img_crop[n_cluster] = lung_img[z_begin:z_end, y_begin:y_end, x_begin:x_end]
        topN_prob3d_all_crop[n_cluster] = prob3d_all[z_begin:z_end, y_begin:y_end, x_begin:x_end]
    
    np.savez(os.path.join(SAVE_DIR, uid+'_cluster_candidates.npz'),
             centroids=topN_centroids,
             lung_img=topN_lung_img_crop,
             prob3d_all=topN_prob3d_all_crop)