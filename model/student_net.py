################################################################################
#
# Authors:
#  (Samuel)Shuming Zhang (uoszhang@gmail.com)
# This is a file about student's network structure(VGG backbone)
#
################################################################################
import tensorflow as tf
from tensorflow import layers as tfl


def student_vgg_block(inputs, filters, kernel_size, name, data_format, training=True,
              batch_normalization=True, kernel_reg=0., **params):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x = tfl.conv2d(inputs, filters, kernel_size, name='conv',
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(kernel_reg),
                       data_format=data_format, **params)
        if batch_normalization:
            x = tfl.batch_normalization(
                    x, training=training, name='bn', fused=True,
                    axis=1 if data_format == 'channels_first' else -1)
    return x


def student_super_vgg_backbone(inputs, **config):
    params_conv = {'padding': 'SAME', 'data_format': 'channels_first',
                   'activation': tf.nn.relu, 'batch_normalization': True,
                   'training': True,
                   'kernel_reg': config.get('kernel_reg', 0.)}
    params_pool = {'padding': 'SAME', 'data_format': 'channels_first'}

    with tf.variable_scope('student_vgg', reuse=tf.AUTO_REUSE):

        x = student_vgg_block(inputs, 64, 3, 'conv1_1', **params_conv)
        x = student_vgg_block(x, 64, 3, 'conv1_2', **params_conv)
        x = tfl.max_pooling2d(x, 2, 2, name='pool1', **params_pool)

        x = student_vgg_block(x, 64, 3, 'conv2_1', **params_conv)
        x = student_vgg_block(x, 64, 3, 'conv2_2', **params_conv)
        x = tfl.max_pooling2d(x, 2, 2, name='pool2', **params_pool)

        x = student_vgg_block(x, 128, 3, 'conv3_1', **params_conv)
        x = student_vgg_block(x, 128, 3, 'conv3_2', **params_conv)
        x = tfl.max_pooling2d(x, 2, 2, name='pool3', **params_pool)

        x = student_vgg_block(x, 128, 3, 'conv4_1', **params_conv)
        x = student_vgg_block(x, 128, 3, 'conv4_2', **params_conv)

    return x