################################################################################
#
# Authors:
#  (Samuel)Shuming Zhang (uoszhang@gmail.com)
# This is the main file used to for knowledge distillation
#
################################################################################
import tensorflow as tf
from datasets import dataset
import argparse
import os
import yaml

from setting import RGB_PATH, EVENT_PATH, WEIGHT_PATH,TENBOARD_PATH
from model.student_net import student_super_vgg_backbone
from model.new_teacher import new_techer_net
from model.student_dec_and_desc_backbone import student_descriptor_head_layer,student_detector_head_layer
from loss_and_metrics.loss_metrics import cal_loss,cal_accuracy,make_prob_nms
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Use GPU 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def show_progress(epoch, feed_dict_train, val_dec_loss, i):

    train_loss = session.run(loss,feed_dict=feed_dict_train)
    msg = "Training Epoch {0}--- iterations: {1}---  Validation Loss: {2:.3f}---Training Loss: {3:.3f}"
    print(msg.format(epoch + 1, i, val_dec_loss,train_loss))

def train(num_iteration):
    global total_iterations
    lr_level = 1
    for i in range(total_iterations, total_iterations + num_iteration):
        data = dataset.read_train_sets(rgb_path, event_path, validation_size, batch_size)
        rgb_batch = data.train._rgb_images
        event_batch = data.train._event_images
        rgb_valid_batch = data.valid._rgb_images
        event_valid_batch = data.valid._event_images

        tr_logits_spv6, tr_prob_spv6, tr_desc_spv6,tr_desc_raw_spv6 = new_techer_net(rgb_batch)

        val_logits_spv6, val_prob_spv6, val_desc_spv6, val_desc_raw_spv6 = new_techer_net(rgb_valid_batch)
        feed_dict_tr = {image: event_batch,logits_spv6: tr_logits_spv6,prob_spv6:tr_prob_spv6,
                        desc_spv6:tr_desc_spv6,desc_raw_spv6:tr_desc_raw_spv6}
        feed_dict_val = {image: event_valid_batch,logits_spv6:val_logits_spv6,prob_spv6:val_prob_spv6,
                         desc_spv6:val_desc_spv6,desc_raw_spv6:val_desc_raw_spv6}
        #Record training data
        session.run(optimizer, feed_dict=feed_dict_tr)
        if i % 10 == 0:
            s = session.run(summ,feed_dict=feed_dict_val)
            writer.add_summary(s, i)
        #Adjust the learning rate
        if i % 100 == 0 and lr_level < 5 and i > 10 :
                lr_level = lr_level + 1
                lr = session.run(learning_rate)
                session.run(tf.assign(learning_rate, 0.1 * lr))
        #Print loss
        if i % 100 == 0:
            val_loss = session.run(loss, feed_dict=feed_dict_val)
            epoch = int(i / 100)
            show_progress(epoch, feed_dict_tr, val_loss, i)

    saver.save(session, './double_net2/double_net.ckpt', global_step=num_iteration)
    total_iterations += num_iteration


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    # config
    batch_size = config['batch_size']
    validation_size = config['validation_size']
    num_filters_1layer = config['num_filters_1layer']
    num_filters_2layer = config['num_filters_2layer']
    iteration = config['iteration']
    # setting
    rgb_path = RGB_PATH
    event_path = EVENT_PATH
    weights_dir = WEIGHT_PATH
    tensorboard_dir = TENBOARD_PATH

    session = tf.Session()
    # placeholder of event and rgb
    image = tf.placeholder(tf.float32, shape=[None, 240, 320, 1], name='event')#for event

    #tensor from sp_v6
    event_image = tf.transpose(image, [0, 3, 1, 2])
    logits_spv6 = tf.placeholder(tf.float32, shape=[None, 30, 40, 65], name='logits_spv6')
    prob_spv6 = tf.placeholder(tf.float32, shape=[None, 240,320], name='prob_spv6')
    desc_spv6 = tf.placeholder(tf.float32, shape=[None, 240,320, 256], name='desc_spv6')
    desc_raw_spv6 = tf.placeholder(tf.float32, shape=[None, 30, 40, 256], name='desc_raw_spv6')

    #feature of event
    feature_event = student_super_vgg_backbone(event_image,**config)

    # lofits and prob of event
    dec_output = student_detector_head_layer(feature_event, **config)
    event_logits = dec_output['logits']
    event_prob = dec_output['prob']
    event_nms_prob = make_prob_nms(event_prob,config)

    # desc_raw and desc of event
    desc_output= student_descriptor_head_layer(feature_event, **config)
    event_desc_raw = desc_output['descriptors_raw']
    event_desc = desc_output['descriptors']
    learning_rate = tf.Variable(1e-6, dtype=tf.float32)
    variables = tf.get_collection(tf.GraphKeys.VARIABLES)

    session.run(tf.global_variables_initializer())
    #setting of loss and optimizer
    loss = cal_loss(event_logits,event_desc_raw, logits_spv6, desc_raw_spv6, **config)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss,var_list=variables)
    accuracy_dec = cal_accuracy(event_nms_prob, prob_spv6, event_desc, desc_spv6)

    writer = tf.summary.FileWriter(tensorboard_dir)
    writer.add_graph(session.graph)
    session.run(tf.global_variables_initializer())

    total_iterations = 0
    saver = tf.train.Saver()

    summ = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    train(num_iteration=iteration)