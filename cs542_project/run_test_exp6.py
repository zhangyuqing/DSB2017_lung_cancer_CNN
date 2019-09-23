import os
import tensorflow as tf
import csv
import numpy as np
import glob

from preprocess import normalize, zero_center

# The data and params
DATA_DIR = './data/preprocessed_add0/'
LABLE_FILE_TEST = './data/stage1_labels_stage1val.csv'
TRAINED_MODEL = './train/exp6/save/00010000'

# The data shape
N, Z, Y, X = 1, 214, 245, 245

sample_list = []
with open(LABLE_FILE_TEST) as f:
    for name, label in csv.reader(f):
        sample_list.append((name, float(label)))
print('number of samples:', len(sample_list))

def get_sample(idx_begin, idx_end):
    inds = range(idx_begin, idx_end)
    data = np.zeros((idx_end-idx_begin, Z, Y, X, 1), np.float32)
    labels = np.zeros((idx_end-idx_begin, 1), np.float32)
    for i, idx in enumerate(inds):
        data[i, ..., 0], labels[i] = read_data(idx)
    return data, labels

def read_data(idx):
    name, label = sample_list[idx]
    d = np.load(os.path.join(DATA_DIR, name+'.npz'))
    pix = zero_center(normalize(d['pix_resampled_add0']))
    return pix, label

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

sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
saver = tf.train.Saver()  # a Saver to save trained models
saver.restore(sess, TRAINED_MODEL)


# Now, start testing
print('Testing started.')
total_loss = 0
correct = 0
for n_iter in range(len(sample_list)):
    print('evaluating %d / %d' % (n_iter+1, len(sample_list)))
    LOADED_DATA, LOADED_GT = get_sample(n_iter, n_iter+1)
    score, loss_value = sess.run((output_final, batch_loss),
                          {data: LOADED_DATA, ground_truth: LOADED_GT})
    correct += ((score > 0) == LOADED_GT[0, 0])

    print('loss = %f' % loss_value)
    total_loss += loss_value

avg_loss = total_loss / len(sample_list)
accuracy = correct / len(sample_list)
print('average loss: %f' % avg_loss)
print('accuracy: %f' % accuracy)
