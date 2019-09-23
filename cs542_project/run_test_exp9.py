import os
import tensorflow as tf
import csv
import numpy as np

################################################################################
# Training setup
################################################################################
# The data and params
DATA_DIR = './data/preprocessed_reshaped/'
DATA_MEAN_FILE = './data/preprocessed_reshaped_mean.npy'
LABLE_FILE_TEST = './data/stage1_labels.csv'
# LABLE_FILE_TEST = './data/stage1_labels_stage1test.csv'

TRAINED_MODEL = './train/exp9_reshape_l2reg/save/00000030'

# The data shape
N, Z, Y, X = 100, 50, 50, 50

################################################################################
# Load data
################################################################################
# Load image and label files
sample_list = []
with open(LABLE_FILE_TEST) as f:
    for name, label in csv.reader(f):
        sample_list.append((name, float(label)))
# number of samples
num_sample = len(sample_list)
print('number of samples:', num_sample)

# # Shuffle the sample list
# np.random.shuffle(sample_list)

# Pre-load all batches into memory
mean_data = np.load(DATA_MEAN_FILE)
all_image_array = np.zeros((num_sample, Z, Y, X, 1), np.float32)
all_label_array = np.zeros((num_sample, 1), np.float32)
for n in range(num_sample):
    print('pre-loading data %d / %d' % (n, num_sample))
    name, label = sample_list[n]
    all_label_array[n, 0] = label
    d = np.load(os.path.join(DATA_DIR, name+'.npz'))
    all_image_array[n, ..., 0] = d['pix_resampled'] - mean_data
    d.close()

################################################################################
# The network
################################################################################
# The placeholder used for feeding inputs during training
data = tf.placeholder(tf.float32, [None, Z, Y, X, 1], name='data')

# our convnet (forward pass)
is_training = False
output = tf.layers.conv3d(data, 32, (4, 4, 4), (1, 1, 1), "SAME", name='conv1', activation=tf.nn.relu)
output = tf.layers.conv3d(output, 32, (4, 4, 4), (1, 1, 1), "SAME", name='conv2')
output = tf.layers.batch_normalization(output, training=is_training, name='batch_norm2')
output = tf.nn.relu(output, name='relu2')
output = tf.layers.max_pooling3d(output, (2, 2, 2), (2, 2, 2), "SAME", name='pool2')

output = tf.layers.conv3d(output, 32, (2, 2, 2), (1, 1, 1), "SAME", name='conv3', activation=tf.nn.relu)
output = tf.layers.conv3d(output, 32, (2, 2, 2), (1, 1, 1), "SAME", name='conv4')
output = tf.layers.batch_normalization(output, training=is_training, name='batch_norm4')
output = tf.nn.relu(output, name='relu4')
output = tf.layers.max_pooling3d(output, (2, 2, 2), (2, 2, 2), "SAME", name='pool4')

output = tf.layers.conv3d(output, 32, (2, 2, 2), (1, 1, 1), "SAME", name='conv5', activation=tf.nn.relu)
output = tf.layers.conv3d(output, 32, (2, 2, 2), (1, 1, 1), "SAME", name='conv6')
output = tf.layers.batch_normalization(output, training=is_training, name='batch_norm6')
output = tf.nn.relu(output, name='relu6')
output = tf.layers.max_pooling3d(output, (2, 2, 2), (2, 2, 2), "SAME", name='pool6')

output = tf.layers.conv3d(output, 40, (2, 2, 2), (1, 1, 1), "SAME", name='conv7', activation=tf.nn.relu)
output = tf.layers.conv3d(output, 40, (2, 2, 2), (1, 1, 1), "SAME", name='conv8')
output = tf.layers.batch_normalization(output, training=is_training, name='batch_norm8')
output = tf.nn.relu(output, name='relu8')
output = tf.layers.max_pooling3d(output, (2, 2, 2), (2, 2, 2), "SAME", name='pool8')

# Flatten the output
output = tf.reshape(output, (-1, np.prod(output.get_shape().as_list()[1:])))

output = tf.layers.dense(output, 128, name='dense9')
output = tf.layers.batch_normalization(output, training=is_training, name='batch_norm9')
output = tf.nn.relu(output, name='relu9')

output = tf.layers.dense(output, 128, name='dense10')
output = tf.layers.batch_normalization(output, training=is_training, name='batch_norm10')
output = tf.nn.relu(output, name='relu10')

output_final = tf.layers.dense(output, 1, name='dense11')

# Loss
ground_truth = tf.placeholder(tf.float32, [None, 1], name='labels')
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ground_truth, logits=output_final)
batch_loss = tf.reduce_mean(loss)

################################################################################
# Training loop
################################################################################
# Now, we've built our network. Let's start a session for training
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
saver = tf.train.Saver(max_to_keep=None)  # a Saver to save trained models
saver.restore(sess, TRAINED_MODEL)

total_loss = 0
correct = 0

num_batch = int(np.ceil(num_sample / N))
for n_batch in range(num_batch):
    idx_begin = n_batch*N
    idx_end = (n_batch+1)*N  # OK if off-the-end
    LOADED_DATA = all_image_array[idx_begin:idx_end]
    LOADED_GT = all_label_array[idx_begin:idx_end]

    score, loss_value = sess.run((output_final, batch_loss),
                                 {data: LOADED_DATA, ground_truth: LOADED_GT})
    
    predicts = (score >= 0)
    correct += np.sum(predicts == LOADED_GT)
    print('batch_loss = %f' % loss_value)
    total_loss += loss_value * len(LOADED_DATA)

avg_loss = total_loss / num_sample
accuracy = correct / num_sample
print('average loss: %f' % avg_loss)
print('accuracy: %f' % accuracy)
