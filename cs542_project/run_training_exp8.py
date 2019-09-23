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
LABLE_FILE_TRAIN = './data/stage1_labels.csv'

NUM_EPOCHES = 1000
exp_name = 'exp8_reshape'
LOG_DIR = './train/%s/tb/' % exp_name
LOG_FILE = './train/%s/log.txt' % exp_name

SAVE_EPOCHES = 10
SAVE_DIR = './train/%s/save/' % exp_name

# The data shape
N, Z, Y, X = 64, 50, 50, 50

# make the directories
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

################################################################################
# Load data
################################################################################
# Load image and label files
sample_list = []
with open(LABLE_FILE_TRAIN) as f:
    for name, label in csv.reader(f):
        sample_list.append((name, float(label)))
# number of samples
num_sample = len(sample_list)
print('number of samples:', num_sample)

# Shuffle the sample list
np.random.shuffle(sample_list)

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

# Set output log file
log_f = open(LOG_FILE, 'w')
def print_and_log(*argv):
    print(*argv, file=log_f, flush=True)
    print(*argv, flush=True)

################################################################################
# The network
################################################################################
# The placeholder used for feeding inputs during training
data = tf.placeholder(tf.float32, [None, Z, Y, X, 1], name='data')

# our convnet (forward pass)
is_training = True
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

# back propagation and parameter updating
train_op = tf.train.AdamOptimizer().minimize(batch_loss)

# other update operations
update_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))

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

# Save the initial parameters
save_path = os.path.join(SAVE_DIR, '%08d' % 0)
saver.save(sess, save_path, write_meta_graph=False)
print_and_log('Model saved to %s' % save_path)
# Now, start training
print_and_log('Training started.')
for n_epoch in range(NUM_EPOCHES):
    num_batch = int(np.ceil(num_sample / N))
    for n_batch in range(num_batch):
        idx_begin = n_batch*N
        idx_end = (n_batch+1)*N  # OK if off-the-end
        LOADED_DATA = all_image_array[idx_begin:idx_end]
        LOADED_GT = all_label_array[idx_begin:idx_end]

        # Training step
        loss_value, _, _, summary = sess.run((batch_loss, train_op, update_op, log_op),
            {data: LOADED_DATA, ground_truth: LOADED_GT})

        # print to output, and save to TensorBoard
        n_iter = n_epoch*num_batch+n_batch
        print_and_log('epoch = %d, batch = %d / %d, iter = %d, loss = %f' \
            % (n_epoch, n_batch, num_batch, n_iter, loss_value))
        log_writer.add_summary(summary, n_iter)

    # save the model every SAVE_ITERS iterations (also save at the beginning)
    if (n_epoch+1) % SAVE_EPOCHES == 0:
        save_path = os.path.join(SAVE_DIR, '%08d' % (n_epoch+1))
        saver.save(sess, save_path)
        print_and_log('Model saved to %s' % save_path)
