################################################################################
#
# Authors:
#  (Samuel)Shuming Zhang (uoszhang@gmail.com)
# This is a file about student's network structure(detector head layer and descriptor head layer)
#
################################################################################
import tensorflow as tf
from tensorflow import layers as tfl
import numpy as np
from .student_net import student_vgg_block

def student_detector_head_layer(inputs,**config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'batch_normalization': True,
                   'training': True,
                   'kernel_reg': config.get('kernel_reg', 0.)}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel

    with tf.variable_scope('student_detector', reuse=tf.AUTO_REUSE):
        x = student_vgg_block(inputs, 256, 3, 'conv1',
                      activation=tf.nn.relu, **params_conv)
        x = student_vgg_block(x, 1+pow(config['grid_size'], 2), 1, 'conv2',
                      activation=None, **params_conv)
        logits = tf.transpose(x, [0, 2, 3, 1], name='event_logits')
        prob = tf.nn.softmax(x, axis=cindex)
        # Strip the extra “no interest point” dustbin
        prob = prob[:, :-1, :, :] if cfirst else prob[:, :, :, :-1]
        prob = tf.depth_to_space(
            prob, config['grid_size'], data_format='NCHW' if cfirst else 'NHWC')
        prob = tf.squeeze(prob, axis=cindex,name='event_prob')

        #tf.summary.scalar('logits1', x)
        #tf.summary.scalar('prob1', prob)
    return {'logits': logits, 'prob': prob}

def student_descriptor_head_layer(inputs,**config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'batch_normalization': True,
                   'training': config['training'],
                   'kernel_reg': config.get('kernel_reg', 0.)}
    cfirst = config['data_format'] == 'channels_first'
    cindex = 1 if cfirst else -1  # index of the channel

    with tf.variable_scope('student_descriptor', reuse=tf.AUTO_REUSE):
        x = student_vgg_block(inputs, 256, 3, 'conv1',
                      activation=tf.nn.relu, **params_conv)
        x = student_vgg_block(x, config['descriptor_size'], 1, 'conv2',
                      activation=None, **params_conv)
        desc_raw = tf.transpose(x, [0, 2, 3, 1], name='event_desc_raw')
        desc = tf.transpose(x, [0, 2, 3, 1]) if cfirst else x
        shape = np.array(desc.get_shape().as_list()[1:3])
        desc = tf.image.resize_bilinear(desc, config['grid_size']*shape)
        desc = tf.transpose(desc, [0, 3, 1, 2]) if cfirst else desc
        desc = tf.nn.l2_normalize(desc, cindex,name='event_desc')

        #tf.summary.scalar('descriptors_raw1', x)
        #tf.summary.scalar('descriptors1', desc)
    return {'descriptors_raw': desc_raw, 'descriptors': desc}