################################################################################
#
# Authors:
#  (Samuel)Shuming Zhang (uoszhang@gmail.com)
# This is a file used to show the qualitative data of model matching
#
#
################################################################################
import argparse
import os
import cv2
import numpy as np
import tensorflow as tf  # noqa: E402
import matplotlib
import yaml
from tqdm import tqdm
matplotlib.use('TkAgg')
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Use GPU 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def extract_SIFT_keypoints_and_descriptors(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(np.squeeze(gray_img), None)

    return kp, desc


def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map,
                                                 keep_k_points=1000):

    def select_k_best(points, k):
        """ Select the k most probable points (and strip their proba).
        points has shape (num_points, 3) where the last coordinate is the proba. """
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        return sorted_prob[-start:, :]

    # Extract keypoints
    keypoints = np.where(keypoint_map > 0)
    prob = keypoint_map[keypoints[0], keypoints[1]]
    keypoints = np.stack([keypoints[0], keypoints[1], prob], axis=-1)
    keys=[]
    keypoints = select_k_best(keypoints, keep_k_points)
    keypoints = keypoints.astype(int)
    for num in keypoints:
        num = num.tolist()
        if num[0] > 10 and num[1] > 10 and num[0] < 230 and num[1] < 310:
            keys.append(num)
    # Get descriptors for keypoints
    keypoints = np.array(keys)
    desc = descriptor_map[keypoints[:, 0], keypoints[:, 1]]

    # Convert from just pts to cv2.KeyPoints
    keypoints = [cv2.KeyPoint(int(p[1]), int(p[0]), 1) for p in keypoints]

    return keypoints, desc


def match_descriptors(kp1, desc1, kp2, desc2):
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches_idx = np.array([m.queryIdx for m in matches])
    m_kp1 = [kp1[idx] for idx in matches_idx]
    matches_idx = np.array([m.trainIdx for m in matches])
    m_kp2 = [kp2[idx] for idx in matches_idx]

    return m_kp1, m_kp2, matches


def compute_homography(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)

    # Estimate the homography between the matches using RANSAC
    H, inliers = cv2.findHomography(matched_pts1[:, [1, 0]],
                                    matched_pts2[:, [1, 0]],
                                    cv2.RANSAC)
    inliers = inliers.flatten()
    return H, inliers


def preprocess_image(img_file, img_size):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, img_size)
    img_orig = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, 2)
    img = img.astype(np.float32)
    img_preprocessed = img / 255.

    return img_preprocessed, img_orig
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Compute the homography \
            between two images with the SuperPoint feature matches.')
    parser.add_argument('config', type=str)
    parser.add_argument('weights_name', type=str)
    parser.add_argument('img1_path', type=str)
    parser.add_argument('img2_path', type=str)
    parser.add_argument('--H', type=int, default=240,
                        help='The height in pixels to resize the images to. \
                                (default: 480)')
    parser.add_argument('--W', type=int, default=320,
                        help='The width in pixels to resize the images to. \
                                (default: 640)')
    parser.add_argument('--k_best', type=int, default=1000,
                        help='Maximum number of keypoints to keep \
                        (default: 1000)')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    weights_name = args.weights_name
    img_size = (args.W, args.H)
    keep_k_best = args.k_best
    save_path = '/home/shuming/result/'
    title = '/home/shuming/compares/night/'
    DND_weights_dir = '/home/shuming/DnD7'
    SP_weights_dir = '/home/shared_data2/eventcamera/exper/saved_models/sp_v6'
    DNDimg_path = '/DND/'
    SPimg_path = '/SP/'
    for i in tqdm(range(1,6)):
        index = str(i)
        DND_img1_file = title + index + DNDimg_path + '1.png'
        DND_img2_file = title + index + DNDimg_path + '2.png'
        SP_img1_file = title + index + SPimg_path + '1.png'
        SP_img2_file = title + index + SPimg_path + '2.png'

        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            tf.saved_model.loader.load(sess,
                                       [tf.saved_model.tag_constants.SERVING],
                                       str(DND_weights_dir))
            DND_input_img_tensor = graph.get_tensor_by_name('event:0')
            DND_output_prob_tensor = graph.get_tensor_by_name('student_detector/event_prob:0')
            DND_output_desc_tensors = graph.get_tensor_by_name('student_descriptor/event_desc:0')

            DND_output_prob_nms_tensor = make_prob_nms(DND_output_prob_tensor,config)
            DND_img1, DND_img1_orig = preprocess_image(DND_img1_file, img_size)
            event_tensor = np.expand_dims(DND_img1, 0)
            DND_out1 = sess.run([DND_output_prob_nms_tensor, DND_output_desc_tensors],
                            feed_dict={DND_input_img_tensor: event_tensor})
            DND_keypoint_map1 = np.squeeze(DND_out1[0])
            DND_keypoint_map1 = DND_keypoint_map1 * 100
            DND_descriptor_map1 = np.squeeze(DND_out1[1])
            DND_kp1, DND_desc1 = extract_superpoint_keypoints_and_descriptors(
                    DND_keypoint_map1, DND_descriptor_map1, keep_k_best)

            DND_img2, DND_img2_orig = preprocess_image(DND_img2_file, img_size)
            DND_out2 = sess.run([DND_output_prob_nms_tensor, DND_output_desc_tensors],
                            feed_dict={DND_input_img_tensor: np.expand_dims(DND_img2, 0)})
            DND_keypoint_map2 = np.squeeze(DND_out2[0])
            DND_keypoint_map2 = DND_keypoint_map2 * 100
            DND_descriptor_map2 = np.squeeze(DND_out2[1])
            DND_kp2, DND_desc2 = extract_superpoint_keypoints_and_descriptors(
                    DND_keypoint_map2, DND_descriptor_map2, keep_k_best)

            # Match and get rid of outliers
            DND_m_kp1, DND_m_kp2, DND_matches = match_descriptors(DND_kp1, DND_desc1, DND_kp2, DND_desc2)
            DND_H, DND_inliers = compute_homography(DND_m_kp1, DND_m_kp2)

            # Draw Double Net Distillation matches
            DND_matches = np.array(DND_matches)[DND_inliers.astype(bool)].tolist()
            DND_matched_img = cv2.drawMatches(DND_img1_orig, DND_kp1, DND_img2_orig, DND_kp2, DND_matches,
                                          None, matchColor=(0, 255, 0),
                                          singlePointColor=(0, 0, 255))
            cv2.imwrite(save_path + index + "/DND_matches.png", DND_matched_img)
            #######################################################
            tf.saved_model.loader.load(sess,
                                       [tf.saved_model.tag_constants.SERVING],
                                       str(SP_weights_dir))

            SP_input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
            SP_output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
            SP_output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')

            SP_img1, SP_img1_orig = preprocess_image(SP_img1_file, img_size)
            SP_out1 = sess.run([SP_output_prob_nms_tensor, SP_output_desc_tensors],
                            feed_dict={SP_input_img_tensor: np.expand_dims(SP_img1, 0)})
            SP_keypoint_map1 = np.squeeze(SP_out1[0])
            SP_descriptor_map1 = np.squeeze(SP_out1[1])
            SP_kp1, SP_desc1 = extract_superpoint_keypoints_and_descriptors(
                SP_keypoint_map1, SP_descriptor_map1, keep_k_best)

            SP_img2, SP_img2_orig = preprocess_image(SP_img2_file, img_size)
            SP_out2 = sess.run([SP_output_prob_nms_tensor, SP_output_desc_tensors],
                            feed_dict={SP_input_img_tensor: np.expand_dims(SP_img2, 0)})
            SP_keypoint_map2 = np.squeeze(SP_out2[0])
            SP_descriptor_map2 = np.squeeze(SP_out2[1])
            SP_kp2, SP_desc2 = extract_superpoint_keypoints_and_descriptors(
                SP_keypoint_map2, SP_descriptor_map2, keep_k_best)

            # Match and get rid of outliers
            SP_m_kp1, SP_m_kp2, SP_matches = match_descriptors(SP_kp1, SP_desc1, SP_kp2, SP_desc2)
            SP_H, SP_inliers = compute_homography(SP_m_kp1, SP_m_kp2)

            # Draw SuperPoint matches
            SP_matches = np.array(SP_matches)[SP_inliers.astype(bool)].tolist()
            SP_matched_img = cv2.drawMatches(SP_img1_orig, SP_kp1, SP_img2_orig, SP_kp2, SP_matches,
                                          None, matchColor=(0, 255, 0),
                                          singlePointColor=(0, 0, 255))

            cv2.imwrite(save_path + index + "/SP_matches.png", SP_matched_img)
