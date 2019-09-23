import os
import tensorflow as tf
import csv
import numpy as np
import glob

from preprocess import normalize, zero_center

# The data and params
DATA_DIR = './data/preprocessed_add0/'
LABLE_FILE_TRAIN = './data/stage1_labels_stage1train.csv'

N_ITERS = 10000
exp_name = 'exp6'
LOG_DIR = './train/%s/tb/' % exp_name
LOG_FILE = './train/%s/log.txt' % exp_name

SAVE_ITERS = 1000
SAVE_DIR = './train/%s/save/' % exp_name

# The data shape
N, Z, Y, X = 1, 214, 245, 245

# make the directories
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

################################################################################
# Load training labels
sample_list = []
with open(LABLE_FILE_TRAIN) as f:
    for name, label in csv.reader(f):
        sample_list.append((name, float(label)))
num_samples = len(sample_list)
print('number of samples:', num_samples)

# Pre-load all the data
data_list = [None]*num_samples
label_list = [None]*num_samples
for idx in range(num_samples):
    print('pre-loading data %d / %d' % (idx+1, len(sample_list)))
    name, label = sample_list[idx]
    d = np.load(os.path.join(DATA_DIR, name + '.npz'))
    pix = zero_center(normalize(d['pix_resampled_add0'])).astype(np.float32)
    data_list[idx] = pix
    label_list[idx] = label
    d.close()

batch_data = np.zeros((N, Z, Y, X, 1), np.float32)
batch_labels = np.zeros((N, 1), np.float32)
def get_random_samples():
    inds = np.random.choice(num_samples, N, replace=False)
    for i, idx in enumerate(inds):
        batch_data[i, ..., 0] = data_list[idx]
        batch_labels[i] = label_list[idx]
    return batch_data, batch_labels

def print_and_log(print_str, file):
    print(print_str, file=file, flush=True)
    print(print_str, flush=True)

################################################################################
# The placeholder used for feeding inputs during training
data = tf.placeholder(tf.float32, [None, Z, Y, X, 1], name='data')

# our convnet (forward pass)
is_training = True
output = tf.layers.conv3d(data, 10, (4, 4, 4), (1, 1, 1), "VALID", activation=tf.nn.relu, name='conv1')
output = tf.layers.conv3d(output, 10, (4, 4, 4), (1, 1, 1), "VALID", activation=tf.nn.relu, name='conv2')
output = tf.layers.batch_normalization(output, training=is_training, name='batch_norm2')
output = tf.layers.max_pooling3d(output, (2, 2, 2), (2, 2, 2), "VALID", name='pool2')

output = tf.layers.conv3d(output, 20, (2, 2, 2), (1, 1, 1), "VALID", activation=tf.nn.relu, name='conv3')
output = tf.layers.conv3d(output, 20, (2, 2, 2), (1, 1, 1), "VALID", activation=tf.nn.relu, name='conv4')
output = tf.layers.batch_normalization(output, training=is_training, name='batch_norm4')
output = tf.layers.max_pooling3d(output, (2, 2, 2), (2, 2, 2), "VALID", name='pool4')

output = tf.layers.conv3d(output, 40, (2, 2, 2), (1, 1, 1), "VALID", activation=tf.nn.relu, name='conv5')
output = tf.layers.conv3d(output, 40, (2, 2, 2), (1, 1, 1), "VALID", activation=tf.nn.relu, name='conv6')
output = tf.layers.batch_normalization(output, training=is_training, name='batch_norm6')
output = tf.layers.max_pooling3d(output, (2, 2, 2), (2, 2, 2), "VALID", name='pool6')

output = tf.reduce_max(output, axis=[1, 2, 3], name='global_maxpool')

output = tf.layers.dense(output, 40, activation=tf.nn.relu, name='dense7')
output = tf.layers.dense(output, 40, activation=tf.nn.relu, name='dense8')
output_final = tf.layers.dense(output, 1, activation=None, name='dense9')

# Loss
ground_truth = tf.placeholder(tf.float32, [None, 1], name='labels')
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ground_truth, logits=output_final)
batch_loss = tf.reduce_mean(loss)

# back propagation and parameter updating
train_op = tf.train.AdamOptimizer().minimize(batch_loss)

################################################################################
# Now, we've built our network. Let's start a session for training
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
sess.run(tf.global_variables_initializer())  # initialize the model parameters
saver = tf.train.Saver()  # a Saver to save trained models

# Set output log file
log_f = open(LOG_FILE, 'w')
# build a TensorBoard summary for visualization
log_writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())
tf.summary.scalar("batch_loss", batch_loss)
log_op = tf.summary.merge_all()

# Now, start training
print_and_log('Training started.', file=log_f)
for n_iter in range(N_ITERS):
    LOADED_DATA, LOADED_GT = get_random_samples()
    loss_value, _, summary = sess.run((batch_loss, train_op, log_op),
                                      {data: LOADED_DATA, ground_truth: LOADED_GT})

    # print to output, and save to TensorBoard
    print_and_log('iter = %d, loss = %f' % (n_iter, loss_value), file=log_f)
    log_writer.add_summary(summary, n_iter)

    # save the model every SAVE_ITERS iterations (also save at the beginning)
    if n_iter == 0 or (n_iter+1) % SAVE_ITERS == 0:
        save_path = os.path.join(SAVE_DIR, '%08d' % (n_iter+1))
        saver.save(sess, save_path)
        print_and_log('Model saved to %s' % save_path, file=log_f)

print_and_log('Optimization done.', file=log_f)
