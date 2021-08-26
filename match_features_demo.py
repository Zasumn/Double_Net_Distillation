import argparse
from pathlib import Path
import yaml
import cv2
import numpy as np
import tensorflow as tf  # noqa: E402
from evaluation.evaluation_setting import SP_weights_dir,DND_weights_dir
  # noqa: E402

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

    keypoints = select_k_best(keypoints, keep_k_points)
    keypoints = keypoints.astype(int)

    # Get descriptors for keypoints
    desc = descriptor_map[keypoints[:, 0], keypoints[:, 1]]

    # Convert from just pts to cv2.KeyPoints
    keypoints = [cv2.KeyPoint(p[1], p[0], 1) for p in keypoints]

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Compute the homography \
            between two images with the SuperPoint feature matches.')
    parser.add_argument('rgb_img1_path', type=str)
    parser.add_argument('rgb_img2_path', type=str)
    parser.add_argument('event_img1_path', type=str)
    parser.add_argument('event_img2_path', type=str)
    parser.add_argument('--H', type=int, default=480,
                        help='The height in pixels to resize the images to. \
                                (default: 480)')
    parser.add_argument('--W', type=int, default=640,
                        help='The width in pixels to resize the images to. \
                                (default: 640)')
    parser.add_argument('--k_best', type=int, default=1000,
                        help='Maximum number of keypoints to keep \
                        (default: 1000)')
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    rgb_img1_file = args.rgb_img1_path
    rgb_img2_file = args.rgb_img2_path
    event_img1_file = args.event_img1_path
    event_img2_file = args.event_img2_path
    img_size = (args.W, args.H)
    keep_k_best = args.k_best


    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tf.saved_model.loader.load(sess,
                                   [tf.saved_model.tag_constants.SERVING],
                                   str(SP_weights_dir))

        input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
        output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
        output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')

        img1, img1_orig = preprocess_image(rgb_img1_file, img_size)
        out1 = sess.run([output_prob_nms_tensor, output_desc_tensors],
                        feed_dict={input_img_tensor: np.expand_dims(img1, 0)})
        keypoint_map1 = np.squeeze(out1[0])
        descriptor_map1 = np.squeeze(out1[1])
        kp1, desc1 = extract_superpoint_keypoints_and_descriptors(
                keypoint_map1, descriptor_map1, keep_k_best)

        img2, img2_orig = preprocess_image(rgb_img2_file, img_size)
        out2 = sess.run([output_prob_nms_tensor, output_desc_tensors],
                        feed_dict={input_img_tensor: np.expand_dims(img2, 0)})
        keypoint_map2 = np.squeeze(out2[0])
        descriptor_map2 = np.squeeze(out2[1])
        kp2, desc2 = extract_superpoint_keypoints_and_descriptors(
                keypoint_map2, descriptor_map2, keep_k_best)

        # Match and get rid of outliers
        m_kp1, m_kp2, matches = match_descriptors(kp1, desc1, kp2, desc2)
        H, inliers = compute_homography(m_kp1, m_kp2)

        # Draw SuperPoint matches
        matches = np.array(matches)[inliers.astype(bool)].tolist()
        matched_img = cv2.drawMatches(img1_orig, kp1, img2_orig, kp2, matches,
                                      None, matchColor=(0, 255, 0),
                                      singlePointColor=(0, 0, 255))

        cv2.imshow("SuperPoint matches", matched_img)

        # Compare SIFT matches
        tf.saved_model.loader.load(sess,
                                   [tf.saved_model.tag_constants.SERVING],
                                   str(DND_weights_dir))
        DND_input_img_tensor = graph.get_tensor_by_name('event:0')
        DND_output_prob_tensor = graph.get_tensor_by_name('student_detector/event_prob:0')
        DND_output_desc_tensors = graph.get_tensor_by_name('student_descriptor/event_desc:0')

        DND_output_prob_nms_tensor = make_prob_nms(DND_output_prob_tensor, config)
        DND_img1, DND_img1_orig = preprocess_image(event_img2_file, img_size)
        event_tensor = np.expand_dims(DND_img1, 0)
        DND_out1 = sess.run([DND_output_prob_nms_tensor, DND_output_desc_tensors],
                            feed_dict={DND_input_img_tensor: event_tensor})
        DND_keypoint_map1 = np.squeeze(DND_out1[0])
        DND_keypoint_map1 = DND_keypoint_map1 * 100
        DND_descriptor_map1 = np.squeeze(DND_out1[1])
        DND_kp1, DND_desc1 = extract_superpoint_keypoints_and_descriptors(
            DND_keypoint_map1, DND_descriptor_map1, keep_k_best)

        DND_img2, DND_img2_orig = preprocess_image(event_img2_file, img_size)
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
        cv2.imshow("SIFT matches", DND_matched_img)

        cv2.waitKey(0)