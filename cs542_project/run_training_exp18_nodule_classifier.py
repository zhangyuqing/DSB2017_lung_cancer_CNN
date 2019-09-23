import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', default='0')
args = parser.parse_args()
import os; os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

import os
import tensorflow as tf
import csv
import numpy as np
import tqdm

################################################################################
# Training setup
################################################################################
# The data and params
DATA_DIR = './data/preprocessed_masks/cluster_candidates_crop32/'
#DATA_MEAN_FILE = './data/preprocessed_reshaped2_mean.npy'
LABLE_FILE_TRAIN = './data/stage1_labels.csv'
LABLE_FILE_TEST = './data/stage1_labels_stage1test.csv'

NUM_EPOCHES = 10000
EXP_NAME = 'exp18_nodule_classifier'
LOG_DIR = './train/%s/tb/' % EXP_NAME
LOG_FILE = './train/%s/log.txt' % EXP_NAME

SAVE_EPOCHES = 1
SAVE_DIR = './train/%s/save/' % EXP_NAME

WEIGHT_DECAY = 1e-3

# The data shape
N = 32
DATA_C, DATA_Z, DATA_Y, DATA_X = 20, 32, 32, 32
#CROP_Z, CROP_Y, CROP_X = 64, 64, 64

# make the directories
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

# Set output log file
log_f = open(LOG_FILE, 'w')
def print_and_log(*argv):
    print(*argv, file=log_f, flush=True)
    print(*argv, flush=True)

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

################################################################################
# Load data
################################################################################

# A. Load training data
# Load image and label files
sample_list_trn = []
with open(LABLE_FILE_TRAIN) as f:
    for name, label in csv.reader(f):
        sample_list_trn.append((name, float(label)))
# number of samples
num_sample_trn = len(sample_list_trn)
print_and_log('number of training samples:', num_sample_trn)

# Shuffle the sample list
np.random.shuffle(sample_list_trn)

# Pre-load all batches into memory
trn_image_array = np.zeros((num_sample_trn, DATA_C, DATA_Z, DATA_Y, DATA_X, 2), np.float32)
trn_label_array = np.zeros((num_sample_trn, 1), np.float32)
for n in tqdm.trange(num_sample_trn):
    name, label = sample_list_trn[n]
    trn_label_array[n, 0] = label
    d = np.load(os.path.join(DATA_DIR, name+'_cluster_candidates.npz'))
    trn_image_array[n, ..., 0] = normalize_and_zero_center(d['lung_img'])
    trn_image_array[n, ..., 1] = d['prob3d']
    d.close()

# compute and subtract mean
mean_data = np.mean(trn_image_array, axis=(0, 1))
trn_image_array -= mean_data

# B. Load test data
# Load image and label files
sample_list_tst = []
with open(LABLE_FILE_TEST) as f:
    for name, label in csv.reader(f):
        sample_list_tst.append((name, float(label)))
# number of samples
num_sample_tst = len(sample_list_tst)
print_and_log('number of test samples:', num_sample_tst)

# Pre-load all batches into memory
tst_image_array = np.zeros((num_sample_tst, DATA_C, DATA_Z, DATA_Y, DATA_X, 2), np.float32)
tst_label_array = np.zeros((num_sample_tst, 1), np.float32)
for n in tqdm.trange(num_sample_tst):
    name, label = sample_list_tst[n]
    tst_label_array[n, 0] = label
    d = np.load(os.path.join(DATA_DIR, name+'_cluster_candidates.npz'))
    tst_image_array[n, ..., 0] = normalize_and_zero_center(d['lung_img'])
    tst_image_array[n, ..., 1] = d['prob3d']
    d.close()

tst_image_array -= mean_data

################################################################################
# The network
################################################################################

# An initializer that keeps the variance the same across layers
def scaling_initializer():
    def _initializer(shape, dtype=None, partition_info=None):
        fan_in = 1.0
        for dim in shape[:-1]:
            fan_in *= dim

        trunc_stddev = np.sqrt(2 / fan_in)
        return tf.truncated_normal(shape, 0., trunc_stddev, dtype)

    return _initializer

kernel_initializer = scaling_initializer()
kernel_regularizer = tf.contrib.layers.l2_regularizer(WEIGHT_DECAY)

def model(input_data, kernel_regularizer, is_training, scope='CNN3D', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        output = tf.layers.conv3d(input_data, 32, (3, 3, 3), (1, 1, 1), "VALID", name='conv1',
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer, activation=tf.nn.relu)
        output = tf.layers.conv3d(output, 32, (3, 3, 3), (1, 1, 1), "VALID", name='conv2',
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer, activation=tf.nn.relu)
        output = tf.layers.max_pooling3d(output, (2, 2, 2), (2, 2, 2), "VALID", name='pool2')

        output = tf.layers.conv3d(output, 32, (3, 3, 3), (1, 1, 1), "VALID", name='conv3',
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer, activation=tf.nn.relu)
        output = tf.layers.conv3d(output, 32, (3, 3, 3), (1, 1, 1), "VALID", name='conv4',
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=kernel_regularizer, activation=tf.nn.relu)
        output = tf.layers.max_pooling3d(output, (2, 2, 2), (2, 2, 2), "VALID", name='pool4')

        print_and_log('shape before fully connected layer:', output.get_shape())

        # Flatten the output
        output = tf.reshape(output, (-1, np.prod(output.get_shape().as_list()[1:])))
        output = tf.layers.dense(output, 64, name='dense5',
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer, activation=tf.nn.relu)
        output = tf.layers.dropout(output, training=is_training, name='drop7')
        output = tf.layers.dense(output, 1, name='dense6',
                                 kernel_initializer=kernel_initializer,
                                 kernel_regularizer=kernel_regularizer)
    return output

# A. Training network
data_trn = tf.placeholder(tf.float32, [None, DATA_Z, DATA_Y, DATA_X, 2], name='data')
output_trn = model(data_trn, kernel_regularizer=kernel_regularizer, is_training=True)

# reverse reshape
output_trn = tf.reshape(output_trn, [-1, DATA_C, 1])
output_trn = tf.reduce_max(output_trn, axis=1)

# Loss
ground_truth_trn = tf.placeholder(tf.float32, [None, 1], name='labels')
loss_trn = tf.nn.sigmoid_cross_entropy_with_logits(labels=ground_truth_trn, logits=output_trn)
batch_loss = tf.reduce_mean(loss_trn)

# Regularization loss
reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

# back propagation and parameter updating
train_op = tf.train.AdamOptimizer().minimize(batch_loss+reg_loss)

# other update operations
update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))

# B. Test network
data_tst = tf.placeholder(tf.float32, [None, DATA_Z, DATA_Y, DATA_X, 2], name='data')
output_tst = model(data_tst, kernel_regularizer=None, is_training=False, reuse=True)

# reverse reshape
output_tst = tf.reshape(output_tst, [-1, DATA_C, 1])
output_tst = tf.reduce_max(output_tst, axis=1)

ground_truth_tst = tf.placeholder(tf.float32, [None, 1], name='labels')
loss_tst = tf.nn.sigmoid_cross_entropy_with_logits(labels=ground_truth_tst, logits=output_tst)

################################################################################
# Training loop
################################################################################
# Now, we've built our network. Let's start a session for training
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())  # initialize the model parameters
saver = tf.train.Saver(max_to_keep=None)  # a Saver to save trained models

# build a TensorBoard summary for visualization
log_writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())
tf.summary.scalar("batch_loss", batch_loss)
log_op = tf.summary.merge_all()

def run_training():
    # Save the initial parameters
    save_path = os.path.join(SAVE_DIR, '%08d' % 0)
    saver.save(sess, save_path, write_meta_graph=False)
    print_and_log('Model saved to %s' % save_path)
    # Now, start training
    print_and_log('Training started.')
    for n_epoch in range(NUM_EPOCHES):
        num_batch = int(np.ceil(num_sample_trn / N))
        for n_batch in range(num_batch):
            idx_begin = n_batch*N
            idx_end = (n_batch+1)*N  # OK if off-the-end
            LOADED_DATA = np.reshape(trn_image_array[idx_begin:idx_end], [-1, DATA_Z, DATA_Y, DATA_X, 2])
            LOADED_GT = trn_label_array[idx_begin:idx_end]

            # Training step
            loss_value, _, _, summary = sess.run((batch_loss, train_op, update_op, log_op),
                {data_trn: LOADED_DATA, ground_truth_trn: LOADED_GT})

            # print to output, and save to TensorBoard
            n_iter = n_epoch*num_batch+n_batch
            print_and_log('epoch = %d, batch = %d / %d, iter = %d, loss = %f' % (n_epoch, n_batch, num_batch, n_iter, loss_value))
            log_writer.add_summary(summary, n_iter)

        # save the model every SAVE_ITERS iterations (also save at the beginning)
        if (n_epoch+1) % SAVE_EPOCHES == 0:
            save_path = os.path.join(SAVE_DIR, '%08d' % (n_epoch+1))
            saver.save(sess, save_path)
            print_and_log('Model saved to %s' % save_path)

            # Test for every snapshot
            run_test('train')
            run_test('test')

def run_test(split):
    total_loss = 0
    correct = 0

    if split == 'train':
        print_and_log('Test on training set started.')
        all_image_array = trn_image_array
        all_label_array = trn_label_array
    elif split == 'test':
        print_and_log('Test on test set started.')
        all_image_array = tst_image_array
        all_label_array = tst_label_array
    else:
        raise ValueError('unknown data split: ' + split)

    num_sample = len(all_image_array)
    num_batch = int(np.ceil(num_sample / N))
    for n_batch in range(num_batch):
        idx_begin = n_batch*N
        idx_end = (n_batch+1)*N  # OK if off-the-end
        LOADED_DATA = np.reshape(all_image_array[idx_begin:idx_end], [-1, DATA_Z, DATA_Y, DATA_X, 2])
        LOADED_GT = all_label_array[idx_begin:idx_end]

        # Training step
        scores, losses = sess.run((output_tst, loss_tst),
                                  {data_tst: LOADED_DATA, ground_truth_tst: LOADED_GT})

        correct += np.sum((scores > 0) == LOADED_GT)
        total_loss += np.sum(losses)

        print_and_log('\tbatch_loss = %f' % np.mean(losses))

    avg_loss = total_loss / num_sample
    accuracy = correct / num_sample
    print_and_log('average loss on %s: %f' % (split, avg_loss))
    print_and_log('accuracy on %s: %f' % (split, accuracy))

if __name__ == '__main__':
    run_training()

run_training()
