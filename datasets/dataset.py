import random

import cv2
import os
import glob
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle

def proc_all_image(rgb_path,event_path,batch_size,validation_size):
    rgb_images_path = sorted(glob.glob(rgb_path))
    event_images_path = sorted(glob.glob(event_path))
    all_image = list(zip(rgb_images_path,event_images_path))
    nmb_sample = int(batch_size*(1+validation_size))
    input_iamge = random.sample(all_image,nmb_sample)
    return input_iamge
def load_train(rgb_path,event_path,batch_size,validation_size):
    event_images = []
    rgb_images = []


    input_image = proc_all_image(rgb_path,event_path,batch_size,validation_size)
    for i in range(int(batch_size*(1+validation_size))):
        rgbimg = cv2.imread(input_image[i][0])
        rgbimg = cv2.cvtColor(rgbimg, cv2.COLOR_BGR2GRAY)
        rgbimg = cv2.resize(rgbimg, (320,240))
        rgbimg = rgbimg.astype(np.float32)
        rgbimg = rgbimg / 255.
        rgbimg = np.expand_dims(rgbimg, axis=2)
        eventimg = cv2.imread(input_image[i][1])
        eventimg = cv2.cvtColor(eventimg, cv2.COLOR_BGR2GRAY)
        eventimg = cv2.resize(eventimg,(320,240))
        eventimg = np.expand_dims(eventimg, axis=2)
        eventimg = eventimg.astype(np.float32)
        eventimg = eventimg / 255.
        rgb_images.append(rgbimg)
        event_images.append(eventimg)
    rgb_images = np.array(rgb_images)

    event_images = np.array(event_images)

    return rgb_images,event_images


class DataSet(object):

  def __init__(self, rgb_images, event_images):
    self._num_examples = rgb_images.shape[0]
    self._rgb_images = rgb_images
    self._event_images = event_images
    self._epochs_done = 0
    self._index_in_epoch = 0

  def _num_examples(self):
    return self._num_examples


  def _rgb_images(self):
    return self._rgb_images


  def _event_images(self):
    return self._event_images


  def _epochs_done(self):
    return self._epochs_done

  def _index_in_epoch(self):
    return self._index_in_epoch

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._rgb_images[start:end], self._event_images[start:end]




def read_train_sets(rgb_path,event_path,validation_size,batch_size):
  class DataSets(object):
    pass
  data_sets = DataSets()

  rgb_images,event_images = load_train(rgb_path, event_path,batch_size,validation_size)

  validation_rgb_images = rgb_images[batch_size:]
  validation_event_images = event_images[batch_size:]


  train_rgb_images = rgb_images[:batch_size]
  train_event_images = event_images[:batch_size]



  data_sets.train = DataSet(train_rgb_images, train_event_images)
  data_sets.valid = DataSet(validation_rgb_images, validation_event_images)

  return data_sets