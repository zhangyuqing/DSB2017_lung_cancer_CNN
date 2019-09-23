import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-b', type=int, default=0)
parser.add_argument('-e', type=int, default=-1)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--epoch', type=int, required=True)
parser.add_argument('--batch_size', type=int, default=10)
args = parser.parse_args()

import os; os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

import numpy as np
import csv
from glob import glob
import tensorflow as tf

import tqdm

from unet import unet

# Parameters
SNAPSHOT_EPOCH = args.epoch
BATCH_SIZE = args.batch_size

DATA_FILE_FILTER = './data/stage1/*'
DATA_DIR = './data/preprocessed_masks/lung_region/'
SAVE_DIR = './data/preprocessed_masks/nodule_prob_epoch_%08d/' % SNAPSHOT_EPOCH

TRAINED_MODEL_AXIS_0 = './train/exp_luna16_unet_nodule_segmentation_axis_0/tfmodel/%08d' % SNAPSHOT_EPOCH
TRAINED_MODEL_AXIS_1 = './train/exp_luna16_unet_nodule_segmentation_axis_1/tfmodel/%08d' % SNAPSHOT_EPOCH
TRAINED_MODEL_AXIS_2 = './train/exp_luna16_unet_nodule_segmentation_axis_2/tfmodel/%08d' % SNAPSHOT_EPOCH


# The network
is_training = False

image_ph = tf.placeholder(tf.float32, [None, None, None, 1])
probs_axis_0 = tf.nn.sigmoid(unet(image_ph, is_training, None, None, 'unet_axiz_0'))
probs_axis_1 = tf.nn.sigmoid(unet(image_ph, is_training, None, None, 'unet_axiz_1'))
probs_axis_2 = tf.nn.sigmoid(unet(image_ph, is_training, None, None, 'unet_axiz_2'))

all_vars = tf.global_variables()
saver_axis_0 = tf.train.Saver([v for v in all_vars if v.name.startswith('unet_axiz_0')])
saver_axis_1 = tf.train.Saver([v for v in all_vars if v.name.startswith('unet_axiz_1')])
saver_axis_2 = tf.train.Saver([v for v in all_vars if v.name.startswith('unet_axiz_2')])

sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
saver_axis_0.restore(sess, TRAINED_MODEL_AXIS_0)
saver_axis_1.restore(sess, TRAINED_MODEL_AXIS_1)
saver_axis_2.restore(sess, TRAINED_MODEL_AXIS_2)

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25
def normalize_and_zero_center(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image = image.astype(np.float32)
    image[image>1] = 1.
    image[image<0] = 0.
    image -= PIXEL_MEAN
    return image

def pad_image(image, stride=16, fill=-PIXEL_MEAN):
    '''
    Pad the image to have length that is a multiplicative of 16,
    by adding HU value of -1024 outside the boundary
    
    This function should be called AFTER normalize_and_zero_center
    '''
    Z, Y, X = image.shape
    Z_pad = int(np.ceil(Z/stride))*stride
    Y_pad = int(np.ceil(Y/stride))*stride
    X_pad = int(np.ceil(X/stride))*stride
    if Z != Z_pad or Y != Y_pad or X != X_pad:
        # The padded value should be the minimum value (-1024 HU)
        # which corresponds to -PIXEL_MEAN after normalization and 0-center
        image_pad = np.zeros((Z_pad, Y_pad, X_pad), image.dtype)
        image_pad[...] = fill
        image_pad[:Z, :Y, :X] = image
    else:
        image_pad = image

    return image_pad

def run_3d_seg_along_axis(image, axis, batch_size=2):
    Z, Y, X = image.shape
    L = image.shape[axis]

    prob3d = np.zeros(image.shape, np.float32)
    
    # split the batches
    num_batches = int(np.ceil(L / batch_size))
    for n_begin in range(0, L, batch_size):
        n_end = min(n_begin+batch_size, L)
        if axis == 0:
            image2d = image[n_begin:n_end, :, :]
            prob2d = sess.run(probs_axis_0, {image_ph: image2d[..., np.newaxis]})[..., 0]
            prob3d[n_begin:n_end, :, :] = prob2d
        elif axis == 1:
            image2d = image[:, n_begin:n_end, :].transpose((1, 0, 2))  # ZYX -> YZX
            prob2d = sess.run(probs_axis_1, {image_ph: image2d[..., np.newaxis]})[..., 0]
            prob3d[:, n_begin:n_end, :] = prob2d.transpose((1, 0, 2))  # YZX -> ZYX
        elif axis == 2:
            image2d = image[:, :, n_begin:n_end].transpose((2, 0, 1))  # ZYX -> XZY
            prob2d = sess.run(probs_axis_2, {image_ph: image2d[..., np.newaxis]})[..., 0]
            prob3d[:, :, n_begin:n_end] = prob2d.transpose((1, 2, 0))  # XZY -> ZYX
        else:
            raise ValueError('Invalid axis ' + str(axis))

    return prob3d


# Load list
file_paths = glob(DATA_FILE_FILTER)
print('number of files:', len(file_paths))
file_paths = file_paths[args.b:args.e]
print('number of files in range %d - %d: %d' % (args.b, args.e, len(file_paths)))

os.makedirs(SAVE_DIR, exist_ok=True)
for n in tqdm.trange(len(file_paths)):
    path = file_paths[n]
    uid = os.path.basename(path)
    
    f = np.load(os.path.join(DATA_DIR, uid+'_lung_region.npz'))
    lung_img, lung_mask = f['lung_img'], f['lung_mask']
    f.close()

    # Apply lung mask
    lung_img[~lung_mask] = -1024
    image = normalize_and_zero_center(lung_img)

    Z, Y, X = image.shape
    image_pad = pad_image(image)
    prob3d_axis_0 = run_3d_seg_along_axis(image_pad, axis=0,
                                          batch_size=BATCH_SIZE)[:Z, :Y, :X]
    prob3d_axis_1 = run_3d_seg_along_axis(image_pad, axis=1,
                                          batch_size=BATCH_SIZE)[:Z, :Y, :X]
    prob3d_axis_2 = run_3d_seg_along_axis(image_pad, axis=2,
                                          batch_size=BATCH_SIZE)[:Z, :Y, :X]
    prob3d = (prob3d_axis_0 + prob3d_axis_1 + prob3d_axis_2) / 3
    assert(prob3d.shape == lung_img.shape)
    
    np.savez(os.path.join(SAVE_DIR, uid+'_nodule_prob.npz'),
             prob3d=prob3d)