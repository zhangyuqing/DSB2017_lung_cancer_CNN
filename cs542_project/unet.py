import tensorflow as tf

def unet(image, is_training, kernel_initializer=None, kernel_regularizer=None, scope='unet', reuse=None,
         output_channels=1):
    kinit = kernel_initializer
    l2reg = kernel_regularizer

    with tf.variable_scope(scope, reuse=reuse):
        conv1 = tf.layers.conv2d(image, 64, (3, 3), padding='SAME', activation=tf.nn.relu,
                                 kernel_initializer=kinit, kernel_regularizer=l2reg, name='conv1_1')
        conv1 = tf.layers.dropout(conv1, 0.2, training=is_training, name='drop1')
        conv1 = tf.layers.conv2d(conv1, 64, (3, 3), padding='SAME', activation=tf.nn.relu,
                                 kernel_initializer=kinit, kernel_regularizer=l2reg, name='conv1_2')
        pool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), name='pool1')

        conv2 = tf.layers.conv2d(pool1, 128, (3, 3), padding='SAME', activation=tf.nn.relu,
                                 kernel_initializer=kinit, kernel_regularizer=l2reg, name='conv2_1')
        conv2 = tf.layers.dropout(conv2, 0.2, training=is_training, name='drop2')
        conv2 = tf.layers.conv2d(conv2, 128, (3, 3), padding='SAME', activation=tf.nn.relu,
                                 kernel_initializer=kinit, kernel_regularizer=l2reg, name='conv2_2')
        pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), name='pool2')

        conv3 = tf.layers.conv2d(pool2, 256, (3, 3), padding='SAME', activation=tf.nn.relu,
                                 kernel_initializer=kinit, kernel_regularizer=l2reg, name='conv3_1')
        conv3 = tf.layers.dropout(conv3, 0.2, training=is_training, name='drop3')
        conv3 = tf.layers.conv2d(conv3, 256, (3, 3), padding='SAME', activation=tf.nn.relu,
                                 kernel_initializer=kinit, kernel_regularizer=l2reg, name='conv3_2')
        pool3 = tf.layers.max_pooling2d(conv3, (2, 2), (2, 2), name='pool3')

        conv4 = tf.layers.conv2d(pool3, 512, (3, 3), padding='SAME', activation=tf.nn.relu,
                                 kernel_initializer=kinit, kernel_regularizer=l2reg, name='conv4_1')
        conv4 = tf.layers.dropout(conv4, 0.2, training=is_training, name='drop4')
        conv4 = tf.layers.conv2d(conv4, 512, (3, 3), padding='SAME', activation=tf.nn.relu,
                                 kernel_initializer=kinit, kernel_regularizer=l2reg, name='conv4_2')
        pool4 = tf.layers.max_pooling2d(conv4, (2, 2), (2, 2), name='pool4')

        conv5 = tf.layers.conv2d(pool4, 1024, (3, 3), padding='SAME', activation=tf.nn.relu,
                                 kernel_initializer=kinit, kernel_regularizer=l2reg, name='conv5_1')
        conv5 = tf.layers.dropout(conv5, 0.2, training=is_training, name='drop5')
        conv5 = tf.layers.conv2d(conv5, 1024, (3, 3), padding='SAME', activation=tf.nn.relu,
                                 kernel_initializer=kinit, kernel_regularizer=l2reg, name='conv5_2')

        up6 = tf.layers.conv2d_transpose(conv5, 512, (2, 2), (2, 2),
                                         kernel_initializer=kinit, kernel_regularizer=l2reg, name='deconv6')
        up6 = tf.concat([up6, conv4], axis=-1)
        conv6 = tf.layers.conv2d(up6, 512, (3, 3), padding='SAME', activation=tf.nn.relu,
                                 kernel_initializer=kinit, kernel_regularizer=l2reg, name='conv6_1')
        conv6 = tf.layers.dropout(conv6, 0.2, training=is_training, name='drop6')
        conv6 = tf.layers.conv2d(conv6, 512, (3, 3), padding='SAME', activation=tf.nn.relu,
                                 kernel_initializer=kinit, kernel_regularizer=l2reg, name='conv6_2')

        up7 = tf.layers.conv2d_transpose(conv6, 256, (2, 2), (2, 2),
                                         kernel_initializer=kinit, kernel_regularizer=l2reg, name='deconv7')
        up7 = tf.concat([up7, conv3], axis=-1)
        conv7 = tf.layers.conv2d(up7, 256, (3, 3), padding='SAME', activation=tf.nn.relu,
                                 kernel_initializer=kinit, kernel_regularizer=l2reg, name='conv7_1')
        conv7 = tf.layers.dropout(conv7, 0.2, training=is_training, name='drop7')
        conv7 = tf.layers.conv2d(conv7, 256, (3, 3), padding='SAME', activation=tf.nn.relu,
                                 kernel_initializer=kinit, kernel_regularizer=l2reg, name='conv7_2')

        up8 = tf.layers.conv2d_transpose(conv7, 128, (2, 2), (2, 2),
                                         kernel_initializer=kinit, kernel_regularizer=l2reg, name='deconv8')
        up8 = tf.concat([up8, conv2], axis=-1)
        conv8 = tf.layers.conv2d(up8, 128, (3, 3), padding='SAME', activation=tf.nn.relu,
                                 kernel_initializer=kinit, kernel_regularizer=l2reg, name='conv8_1')
        conv8 = tf.layers.dropout(conv8, 0.2, training=is_training, name='drop8')
        conv8 = tf.layers.conv2d(conv8, 128, (3, 3), padding='SAME', activation=tf.nn.relu,
                                 kernel_initializer=kinit, kernel_regularizer=l2reg, name='conv8_2')

        up9 = tf.layers.conv2d_transpose(conv8, 64, (2, 2), (2, 2),
                                         kernel_initializer=kinit, kernel_regularizer=l2reg, name='deconv9')
        up9 = tf.concat([up9, conv1], axis=-1)
        conv9 = tf.layers.conv2d(up9, 64, (3, 3), padding='SAME', activation=tf.nn.relu,
                                 kernel_initializer=kinit, kernel_regularizer=l2reg, name='conv9_1')
        conv9 = tf.layers.dropout(conv9, 0.2, training=is_training, name='drop9')
        conv9 = tf.layers.conv2d(conv9, 64, (3, 3), padding='SAME', activation=tf.nn.relu,
                                 kernel_initializer=kinit, kernel_regularizer=l2reg, name='conv9_2')

        conv10 = tf.layers.conv2d(conv9, output_channels, (1, 1),
                                  kernel_initializer=kinit, kernel_regularizer=l2reg, name='conv10')
        
    return conv10
