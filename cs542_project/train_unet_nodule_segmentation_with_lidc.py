import os

import numpy as np
import tensorflow as tf
import csv

from unet import unet

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default='0')
parser.add_argument('--axis', default='0')
args = parser.parse_args()

import os; os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

# Parameters
H_CROP, W_CROP = 128, 128

WEIGHT_DECAY = 5e-4
SCAN_AXIS = int(args.axis)  # 0 for Z, 1 for Y, 2 for X
NET_NAME = 'unet_axiz_%d' % SCAN_AXIS

POS_WEIGHT = 100
NEG_WEIGHT = 0.5

BATCH_SIZE = 20
SAVE_INTERVAL = 5
MAX_EPOCH = 150

ANN_FILE = './data_luna16/annotations.csv'
DATA_DIR = './data_luna16/preprocessed_masks/unet_train_with_lidc/'

EXP_NAME = 'exp_luna16_unet_nodule_segmentation_with_lidc_axis_%d' % SCAN_AXIS

# The network
is_training = True

image_ph = tf.placeholder(tf.float32, [None, H_CROP, W_CROP, 1])
kinit = tf.contrib.layers.variance_scaling_initializer()
l2reg = tf.contrib.layers.l2_regularizer(WEIGHT_DECAY)
scores = unet(image_ph, is_training, kinit, l2reg, NET_NAME, output_channels=2)

# Loss
label_ph = tf.placeholder(tf.float32, [None, H_CROP, W_CROP, 2])
loss_cls = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=scores,
    targets=label_ph, pos_weight=POS_WEIGHT/NEG_WEIGHT) * NEG_WEIGHT)

loss_reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
train_step = tf.train.AdamOptimizer().minimize(loss_cls + loss_reg)

save_dir = os.path.join('./train', EXP_NAME, 'tfmodel')
save_path = os.path.join(save_dir, '%08d')
os.makedirs(save_dir, exist_ok=True)
saver = tf.train.Saver(max_to_keep=None)

log_dir = os.path.join('./train', EXP_NAME, 'tb')
os.makedirs(log_dir, exist_ok=True)
log_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
tf.summary.scalar('weighted_loss', loss_cls)
log_step = tf.summary.merge_all()

sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())

def load_luna16_annotations(ann_file):
    '''
    Load the LUNA16 annotations into a dictionary:
        {<uid>: [[X1, Y1, Z1, D1], [X2, Y2, Z2, D2], ...]}
        where the key is the uid string like
        '1.3.6.1.4.1.14519.5.2.1.6279.6001.122621219961396951727742490470'
        and the value is a list of length-4 list, containing X, Y, Z coordinate and diameter
        of each nodule in the patient, represented in world coordinate
    '''
    sample_dict = {}
    with open(ann_file) as f:
        for uid, X, Y, Z, D in csv.reader(f):
            # skip the first line
            if uid == 'seriesuid':
                continue

            if uid not in sample_dict:
                sample_dict[uid] = [[float(X), float(Y), float(Z), float(D)]]
            else:
                sample_dict[uid].append([float(X), float(Y), float(Z), float(D)])

    return sample_dict

def random_crop_along_axis(image, label, idx, axis, h_crop, w_crop):
    if axis == 0:
        image2d = image[idx, :, :]
        label2d = label[idx, :, :]
    elif axis == 1:
        image2d = image[:, idx, :]
        label2d = label[:, idx, :]
    elif axis == 2:
        image2d = image[:, :, idx]
        label2d = label[:, :, idx]
    else:
        raise ValueError('Invalid axis ' + str(axis))

    h_im, w_im = image2d.shape
    h_begin = np.random.randint(h_im-h_crop)
    w_begin = np.random.randint(w_im-w_crop)
    h_end = h_begin+h_crop
    w_end = w_begin+w_crop

    image2d = image2d[h_begin:h_end, w_begin:w_end]
    label2d = label2d[h_begin:h_end, w_begin:w_end]
    return image2d, label2d

def get_random_crops(image, label, axis, h_crop, w_crop, batch_size):
    # find the index of non-zero scans along the specified axis
    scan_inds = np.nonzero(label)[axis]
    if len(scan_inds) > batch_size:
        print('keeping %d scans out of %d to fit in batch' % (batch_size, len(scan_inds)))
        scan_inds = np.random.choice(scan_inds, batch_size, replace=False)
    num_scans = len(scan_inds)

    image_crops = np.zeros((num_scans, h_crop, w_crop), image.dtype)
    label_crops = np.zeros((num_scans, h_crop, w_crop), label.dtype)

    for n_idx, idx in enumerate(scan_inds):
        image_crops[n_idx], label_crops[n_idx] = random_crop_along_axis(image, label, idx, axis, h_crop, w_crop)
    return image_crops, label_crops

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

uid2anns = load_luna16_annotations(ANN_FILE)
uid_list = list(uid2anns.keys())
np.random.shuffle(uid_list)

num_samples = len(uid_list)
pos_count = 0

saver.save(sess, save_path % 0, write_meta_graph=False)
print('snapshot saved to ' + save_path % 0)
for n_epoch in range(MAX_EPOCH):
    for n_batch, uid in enumerate(uid_list):
        n_iter = n_epoch*num_samples+n_batch
        print('epoch = %d, batch = %d, iter = %d' % (n_epoch, n_batch, n_iter))

        d = np.load(os.path.join(DATA_DIR, uid+'_unet_train_with_lidc.npz'))
        image_crops, label_crops = get_random_crops(d['image'], d['label'], SCAN_AXIS, H_CROP, W_CROP, BATCH_SIZE)
        # The nodule label in the 1st channel
        # The malignancy label is in the 2nd channel
        label_crops = np.stack([(label_crops % 2), (label_crops // 2)], axis=-1).astype(np.float32)  
        image_crops = normalize_and_zero_center(image_crops)
        d.close()

        pos_count += np.mean(label_crops)
        print('fraction of positive pixels: %f' % (pos_count / (n_iter+1)))
        loss_cls_value, _, summary = sess.run((loss_cls, train_step, log_step),
                                              {image_ph: image_crops[..., np.newaxis],
                                               label_ph: label_crops})
        log_writer.add_summary(summary, n_iter)
        print('\tloss: %f' % loss_cls_value)

    if (n_epoch+1) % SAVE_INTERVAL == 0:
        saver.save(sess, save_path % (n_epoch+1), write_meta_graph=False)
        print('snapshot saved to ' + save_path % (n_epoch+1))
