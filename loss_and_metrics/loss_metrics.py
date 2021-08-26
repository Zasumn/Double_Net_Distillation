################################################################################
#
# Authors:
#  (Samuel)Shuming Zhang (uoszhang@gmail.com)
# This is a file about loss function
#
################################################################################
import tensorflow as tf

def box_nms(prob, size, iou=0.1, min_prob=0.001, keep_top_k=0):
    with tf.name_scope('box_nms'):
        pts = tf.to_float(tf.where(tf.greater_equal(prob, min_prob)))
        size = tf.constant(size/2.)
        boxes = tf.concat([pts-size, pts+size], axis=1)
        scores = tf.gather_nd(prob, tf.to_int32(pts))
        with tf.device('/cpu:0'):
            indices = tf.image.non_max_suppression(
                    boxes, scores, tf.shape(boxes)[0], iou)
        pts = tf.gather(pts, indices)
        scores = tf.gather(scores, indices)
        if keep_top_k:
            k = tf.minimum(tf.shape(scores)[0], tf.constant(keep_top_k))  # when fewer
            scores, indices = tf.nn.top_k(scores, k)
            pts = tf.gather(pts, indices)
        prob = tf.scatter_nd(tf.to_int32(pts), scores, tf.shape(prob))
    return prob

def spatial_nms(prob, size):
    with tf.name_scope('spatial_nms'):
        prob = tf.expand_dims(tf.expand_dims(prob, axis=0), axis=-1)
        pooled = tf.nn.max_pool(
                prob, ksize=[1, size, size, 1], strides=[1, 1, 1, 1], padding='SAME')
        prob = tf.where(tf.equal(prob, pooled), prob, tf.zeros_like(prob))
        return tf.squeeze(prob)

def make_prob_nms(prob,config):
    if config['nms']:
        if config['box_nms']:
            prob = tf.map_fn(lambda p: box_nms(p, config['nms'], keep_top_k=config['top_k']), prob)
            prob = tf.add(prob,0,name='event_nms_prob')
        else:
            prob = tf.map_fn(lambda p: spatial_nms(p, config['nms']), prob)
        #tf.summary.scalar('nms_prob1', prob)
    return prob

def cal_accuracy(prob_nms, keypoint_map,desc,descriptor_map):
    prob_nms = tf.multiply(prob_nms,255)
    keypoint_map = tf.multiply(keypoint_map, 255)
    prob_nms = tf.to_int32(prob_nms)
    keypoint_map= tf.to_int32(keypoint_map)
    desc = tf.multiply(desc, 255)
    descriptor_map = tf.multiply(descriptor_map, 255)
    desc= tf.to_int32(desc)
    desc_map= tf.to_int32(descriptor_map)

    correct_count_dec = tf.equal(prob_nms, keypoint_map)
    correct_count_dec = tf.cast(correct_count_dec, tf.float32)
    accuracy_dec = tf.reduce_mean(correct_count_dec)
    correct_count_desc = tf.equal(desc, desc_map)
    correct_count_desc = tf.cast(correct_count_desc, tf.float32)
    accuracy_desc = tf.reduce_mean(correct_count_desc)
    tf.summary.scalar('accuracy_dec', accuracy_dec)
    tf.summary.scalar('accuracy_desc', accuracy_desc)
    accuracy_all = (accuracy_desc + accuracy_dec) /2
    tf.summary.scalar('accuracy_all', accuracy_all)
    return accuracy_all




def cal_loss(event_logits,event_desc_raw, rgb_logits, rgb_desc_raw,**config):
    def pro_loss(input_loss):
        output_loss = tf.transpose(input_loss, [0, 3, 1, 2])

        return output_loss
    # RGB output
    logits1 = pro_loss(rgb_logits)
    descriptor1 = pro_loss(rgb_desc_raw)
    # event output
    logits2 = pro_loss(event_logits)
    descriptor2 = pro_loss(event_desc_raw)

    detector_loss = tf.nn.l2_loss(tf.subtract(logits2, logits1))
    descriptor_loss = tf.nn.l2_loss(tf.subtract(descriptor2, descriptor1))
    tf.summary.scalar('detector_loss', detector_loss)
    tf.summary.scalar('descriptors_loss', descriptor_loss)
    loss = detector_loss + config['lambda_loss'] * descriptor_loss
    tf.summary.scalar('total_loss', loss)
    return loss

