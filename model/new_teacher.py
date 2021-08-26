################################################################################
#
# Authors:
#  (Samuel)Shuming Zhang (uoszhang@gmail.com)
# This is a file used to extract the features from the teacher model
#
################################################################################
import tensorflow as tf
import numpy as np
def new_techer_net(rgb_img_batch):
    weights_dir = '/home/shared_data2/eventcamera/exper/saved_models/sp_v6'
    logits_spv6_batch = []
    prob_spv6_batch = []
    desc_spv6_batch = []
    desc_raw_spv6_batch = []

    for i in range(rgb_img_batch.shape[0]):
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:

            rgb_img = rgb_img_batch[i, :, :, :]
            rgb_img = np.expand_dims(rgb_img, 0)
            tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING],str(weights_dir))
            input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
            output_det_logits = graph.get_tensor_by_name('superpoint/logits:0')
            output_desc_raw = graph.get_tensor_by_name('superpoint/descriptors_raw:0')
            output_prob_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
            output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')

            prob_spv6 = sess.run(output_prob_tensor, feed_dict={input_img_tensor: rgb_img})
            prob_spv6 = np.squeeze(prob_spv6,0)
            desc_spv6 = sess.run(output_desc_tensors, feed_dict={input_img_tensor: rgb_img})
            desc_spv6 = np.squeeze(desc_spv6, 0)
            logits_spv6 = sess.run(output_det_logits, feed_dict={input_img_tensor: rgb_img})
            logits_spv6 = np.squeeze(logits_spv6, 0)
            desc_raw_spv6 = sess.run(output_desc_raw, feed_dict={input_img_tensor: rgb_img })
            desc_raw_spv6 = np.squeeze(desc_raw_spv6, 0)
            logits_spv6_batch.append(logits_spv6)
            prob_spv6_batch.append(prob_spv6)
            desc_spv6_batch.append(desc_spv6)
            desc_raw_spv6_batch.append(desc_raw_spv6)
    logits_spv6_batch = np.array(logits_spv6_batch)
    prob_spv6_batch = np.array(prob_spv6_batch)
    desc_spv6_batch = np.array(desc_spv6_batch)
    desc_raw_spv6_batch = np.array(desc_raw_spv6_batch)
    return logits_spv6_batch, prob_spv6_batch, desc_spv6_batch, desc_raw_spv6_batch
    #return logits_spv6,prob_spv6,desc_spv6,desc_raw_spv6

