################################################################################
#
# Authors:
#  (Samuel)Shuming Zhang (uoszhang@gmail.com)
# This is a file about model's evaluations
#
################################################################################
import argparse
import yaml
import cv2
import numpy as np
import tensorflow as tf  # noqa: E402
import matplotlib
import os
import re
from pyquaternion import Quaternion
from evaluation_setting import INS_PATH,TP_PATH,RGB_PATH,EVENT_PATH,DND_weights_dir,SP_weights_dir
import interpolate_poses as IP
import glob
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 使用 GPU 1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#Get paths from  evaluation_setting.py

matplotlib.use('TkAgg')
K = np.array([[400.000000, 0.0000000e+00, 400.000000],
              [0.0000000e+00, 508.222931, 498.187378],
              [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])


delay = 100
samp = 1000
index = 0
nr, nt, nro, nto, nrs, nts = 0, 0, 0, 0, 0, 0
nrose,ntose,ntdr,nrdr = 0,0,0,0

with open(TP_PATH, "r") as f:
    tdata = f.readlines()


def proc_all_image(rgb_path,event_path):
    rgb_images_path = sorted(glob.glob(rgb_path))
    event_images_path = sorted(glob.glob(event_path))
    all_image = list(zip(rgb_images_path,event_images_path))
    return all_image

def extract_SIFT_keypoints_and_descriptors(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(np.squeeze(gray_img), None)
    return kp, desc


def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map,event,
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
    if event :
        keypoints = [cv2.KeyPoint(int(p[1] * 3.175), int(p[0] * 4.15), 1) for p in keypoints]
    else:
        keypoints = [cv2.KeyPoint(p[1], p[0], 1) for p in keypoints]

    return keypoints, desc


def match_descriptors(kp1, desc1, kp2, desc2,event):
    # Match the keypoints with the warped_keypoints with nearest neighbor search
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    m_kp1=[]
    m_kp2=[]
    new_matches=[]
    if event:
        for m in matches:
            qu_idx = m.queryIdx
            tr_idx = m.trainIdx
            if kp1[qu_idx].pt[0] != 0 and kp1[qu_idx].pt[1] != 0 and kp2[tr_idx].pt[0] != 0 and kp2[tr_idx].pt[1] != 0:
                m_kp1.append(kp1[qu_idx])
                m_kp2.append(kp2[tr_idx])
                new_matches.append(m)
        return m_kp1, m_kp2, new_matches
    else:
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

def preprocess_image(img_file, img_size,event):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (1016, 996), interpolation=cv2.INTER_LANCZOS4)
    img_orig = img.copy()
    if event:
        img = cv2.resize(img, img_size)
        cv2.rectangle(img, (0,0), (240,320), (0,0,0), 8)
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
    return prob

def get_matches(input_tenser,prob_tensor,desc_tensor,img1,img2,event):
    if event :
        prob_nms_tensor = make_prob_nms(prob_tensor, config)
    else:
        prob_nms_tensor = prob_tensor
    out1 = sess.run([prob_nms_tensor, desc_tensor],
                    feed_dict={input_tenser: np.expand_dims(img1, 0)})
    keypoint_map1 = np.squeeze(out1[0])
    descriptor_map1 = np.squeeze(out1[1])
    if event:
        keypoint_map1 = keypoint_map1 * 100
        descriptor_map1 = descriptor_map1 *10
        #keypoint_map1 = cv2.resize(keypoint_map1, (1016, 996), interpolation=cv2.INTER_LANCZOS4)
        #descriptor_map1 = cv2.resize(descriptor_map1, (1016, 996), interpolation=cv2.INTER_LANCZOS4)
    kp1, desc1 = extract_superpoint_keypoints_and_descriptors(
        keypoint_map1, descriptor_map1,event, keep_k_best)
    out2 = sess.run([prob_nms_tensor, desc_tensor],
                    feed_dict={input_tenser: np.expand_dims(img2, 0)})
    keypoint_map2 = np.squeeze(out2[0])
    descriptor_map2 = np.squeeze(out2[1])
    if event:
        keypoint_map2 = keypoint_map2 * 100
        descriptor_map2 = descriptor_map2 * 10
        #keypoint_map2 = cv2.resize(keypoint_map2, (1016, 996), interpolation=cv2.INTER_LANCZOS4)
        #descriptor_map2 = cv2.resize(descriptor_map2, (1016, 996), interpolation=cv2.INTER_LANCZOS4)
    kp2, desc2 = extract_superpoint_keypoints_and_descriptors(
        keypoint_map2, descriptor_map2,event, keep_k_best)
    # Match and get rid of outliers
    m_kp1, m_kp2, matches = match_descriptors(kp1, desc1, kp2, desc2,event)
    H, inliers = compute_homography(m_kp1, m_kp2)
    # Draw SuperPoint matches
    matches = np.array(matches)[inliers.astype(bool)].tolist()
    return m_kp1,m_kp2,matches
def compute_fundamental(matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)
    F, mask = cv2.findFundamentalMat(matched_pts1, matched_pts2, cv2.FM_RANSAC)
    return F, mask


def compute_essential(matched_kp1, matched_kp2, cam='right'):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)
    if (cam == 'right'):
        E, mask = cv2.findEssentialMat(matched_pts1, matched_pts2, K, method=cv2.RANSAC, threshold=1.5, prob=0.999)
    if (cam == 'left'):
        E, mask = cv2.findEssentialMat(matched_pts1, matched_pts2, K, method=cv2.RANSAC, threshold=1.5, prob=0.999)

    return E, mask


def recoverPose(E, matched_kp1, matched_kp2):
    matched_pts1 = cv2.KeyPoint_convert(matched_kp1)
    matched_pts2 = cv2.KeyPoint_convert(matched_kp2)
    points, R_est, t_est, mask_pose = cv2.recoverPose(E, matched_pts1, matched_pts2, K)
    return R_est, t_est


def SE3_eval(R, t, RG, TG):
    t = t.reshape(3, )
    TG = np.flipud(TG.reshape(3, ))
    print(t)
    print(TG)
    lr = np.sqrt(R.dot(R))
    lt = np.sqrt(t.dot(t))
    lrg = np.sqrt(RG.dot(RG))
    ltg = np.sqrt(TG.dot(TG))
    cosr = R.dot(RG) / (lr * lrg)
    cost = t.dot(TG) / (lt * ltg)

    if (cosr < 0): cosr = 0
    if (cost < 0): cost = 0
    if (cosr > 1): cosr = 1
    if (cost > 1): cost = 1
    return cosr, cost


def normalization(data):
    return np.array(data / np.sqrt(data.dot(data.T)))

def calc_matrix_and_recover(m_kp1, m_kp2, matches):
    F, mask_f = compute_fundamental(m_kp1, m_kp2)
    E, mask_e = compute_essential(m_kp1, m_kp2)
    R, t = recoverPose(E, m_kp1, m_kp2)
    return R, t, F, E


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute the homography \
            between two images with the SuperPoint feature matches.')
    parser.add_argument('config', type=str)
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
    img_size = (args.W, args.H)
    keep_k_best = args.k_best
    paths = proc_all_image(RGB_PATH,EVENT_PATH)
    paths.sort()
    base_rgb = ''
    base_event = ''
    SP_acc = 0
    DND_acc = 0
    SIFT_acc = 0
    #for img_path in paths:
    for i in range(358):
        ##number = int(str(img_path[0])[-9:-4])
        pose_path = paths[i+1]
        base_path = paths[i]

        base_rgb = base_path[0]
        base_event = base_path[1]
        pose_rgb = pose_path[0]
        pose_event = pose_path[1]
        print('==>Successfully load picture pair {} and {}'.format(base_rgb,base_event))
        index += 1
        if index > 1 :

            base = int(re.sub("\D", "", base_rgb))
            pose = int(re.sub("\D", "", pose_rgb))
            pose_timestamps = []
            pose_timestamps.append(pose)
            origin_timestamp = base
            #pose_timestamps = [int(tdata[int(number)][:-2])]
            #origin_timestamp = int(tdata[int(number_base)][:-2])
            GT = IP.interpolate_ins_poses(INS_PATH, pose_timestamps, origin_timestamp, use_rtk=False)
            print('Ground truth from ins file is')
            RG = np.matrix([[GT[0][0, 0], GT[0][0, 1], GT[0][0, 2]], [GT[0][1, 0], GT[0][1, 1], GT[0][1, 2]],
                            [GT[0][2, 0], GT[0][2, 1], GT[0][2, 2]]])
            TG = np.matrix([GT[0][0, 3], GT[0][1, 3], GT[0][2, 3]])
            print('Translation Ground truth from ins file is')
            print(RG)
            print('Rotation Ground truth from ins file is')
            print(TG)
            R_quat = list(Quaternion(matrix=RG))
            R_quat = np.array(R_quat)
            T_norm = normalization(TG)
            print(R_quat)
            print(T_norm)

            #open a graph to operate
            graph = tf.Graph()
            with tf.Session(graph=graph) as sess:
                ###process image###
                rgb_img1, rgb_img1_orig = preprocess_image(base_rgb, img_size,False)#img_path[0]--->rgb
                rgb_img2, rgb_img2_orig = preprocess_image(pose_rgb, img_size,False)
                ###process image###
                ###calculate Superpoint###
                tf.saved_model.loader.load(sess,
                                           [tf.saved_model.tag_constants.SERVING],
                                           str(SP_weights_dir))
                input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
                output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
                output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')

                m_kp1_sp, m_kp2_sp, matches_sp = get_matches(input_img_tensor,output_prob_nms_tensor,output_desc_tensors,rgb_img1,rgb_img2,False)
                print('====>The result of Superpoint!<====')
                Ro, to, Fo, Eo = calc_matrix_and_recover(m_kp1_sp, m_kp2_sp, matches_sp)
                print('found {} matches'.format(len(matches_sp)))
                print('rotation matrix estimated is')
                print(Ro)
                print('translation vector estimated is')
                print(to)
                Ro = list(Quaternion(matrix=Ro))
                Ro = np.array(Ro)
                lRs, lts = SE3_eval(Ro, to, R_quat, T_norm)
                nro += lRs
                nto += lts
                print('average R acc is:{}, average t acc is:{}'.format(nro / index, nto / index))

                ###calculate Superpoint###

                ###calculate double net distillation###
                event_img1, event_img1_orig = preprocess_image(base_event, img_size,True)#img_path[1]--->event
                event_img2, event_img2_orig = preprocess_image(pose_event, img_size,True)
                tf.saved_model.loader.load(sess,
                                           [tf.saved_model.tag_constants.SERVING],
                                           str(DND_weights_dir))
                input_event_tensor = graph.get_tensor_by_name('event:0')
                output_event_prob_tensor = graph.get_tensor_by_name('student_detector/event_prob:0')
                output_event_desc_tensors = graph.get_tensor_by_name('student_descriptor/event_desc:0')

                m_kp1_dnd, m_kp2_dnd, matches_dnd = get_matches(input_event_tensor,output_event_prob_tensor,output_event_desc_tensors,event_img1,event_img2,True)
                print('====>The result of DND!<====')
                R, t, F, E = calc_matrix_and_recover(m_kp1_dnd, m_kp2_dnd, matches_dnd)
                print('found {} matches'.format(len(matches_dnd)))
                print('rotation matrix estimated is')
                print(R)
                print('translation vector estimated is')
                print(t)
                R = list(Quaternion(matrix=R))
                R = np.array(R)
                lRe, lte = SE3_eval(R, t, R_quat, T_norm)
                nr += lRe
                nt += lte
                print('average R acc is:{}, average t acc is:{}'.format(nr / index, nt / index))

                ###calculate double net distillation###
                ###calculate SIFT###
                sift_kp1, sift_desc1 = extract_SIFT_keypoints_and_descriptors(event_img1_orig)
                sift_kp2, sift_desc2 = extract_SIFT_keypoints_and_descriptors(event_img2_orig)

                m_kp1_sift, m_kp2_sift, matches_sift = match_descriptors(
                    sift_kp1, sift_desc1, sift_kp2, sift_desc2,False)
                sift_H, sift_inliers = compute_homography(m_kp1_sift, m_kp2_sift)

                sift_matches = np.array(matches_sift)[sift_inliers.astype(bool)].tolist()
                print('====>The result of Sift!<====')
                Rs, ts, Fs, Es = calc_matrix_and_recover(m_kp1_sift, m_kp2_sift, sift_matches)
                print('found {} matches'.format(len(sift_matches)))
                print('rotation matrix estimated is')
                print(Rs)
                print('translation vector estimated is')
                print(ts)
                Rs = list(Quaternion(matrix=Rs))
                Rs = np.array(Rs)
                lR, lt = SE3_eval(Rs, ts, R_quat, T_norm)
                nrs += lR
                nts += lt
                print('average R acc is:{}, average t acc is:{}'.format(nrs / index, nts / index))
                ###calculate SIFT###
