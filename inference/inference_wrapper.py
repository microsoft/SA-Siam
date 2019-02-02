#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily          Huazhong University of Science and Technology
# Copyright © 2018 Anfeng He     Microsoft Research Asia. University of Science and Technology of China.
# Copyright (c) Microsoft. All rights reserved.
#
# Distributed under terms of the MIT license.

"""Model Wrapper class for performing inference with a SiameseModel"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import logging
logging.getLogger().setLevel(logging.INFO)
import os
import os.path as osp

import numpy as np
import tensorflow as tf

from embeddings.sa_siam import sa_siam_arg_scope, sa_siam
from utils.infer_utils import get_exemplar_images
from utils.misc_utils import get_center, get, shape_of

slim = tf.contrib.slim


class InferenceWrapper():
  """Model wrapper class for performing inference with a siamese model."""

  def __init__(self):
    self.image = None
    self.target_bbox_feed = None
    self.search_images = None
    self.embeds = None
    self.templates = None
    self.init = None
    self.model_config = None
    self.track_config = None
    self.response_up = None

  def build_graph_from_config(self, model_config, track_config, checkpoint_path):
    """Build the inference graph and return a restore function."""
    self.build_model(model_config, track_config)
    ema = tf.train.ExponentialMovingAverage(0)
    variables_to_restore = ema.variables_to_restore(moving_avg_variables=[])

    # Filter out State variables
    variables_to_restore_filterd = {}
    for key, value in variables_to_restore.items():
      if key.split('/')[1] != 'State':
        variables_to_restore_filterd[key] = value

    saver = tf.train.Saver(variables_to_restore_filterd)

    if osp.isdir(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
      if not checkpoint_path:
        raise ValueError("No checkpoint file found in: {}".format(checkpoint_path))

    def _restore_fn(sess):
      logging.info("Loading model from checkpoint: %s", checkpoint_path)
      saver.restore(sess, checkpoint_path)
      logging.info("Successfully loaded checkpoint: %s", os.path.basename(checkpoint_path))

    return _restore_fn

  def build_model(self, model_config, track_config):
    self.model_config = model_config
    self.track_config = track_config

    self.build_inputs()
    self.build_search_images()
    self.build_template()
    self.build_detection()
    self.build_upsample()
    self.dumb_op = tf.no_op('dumb_operation')

  def build_inputs(self):
    filename = tf.placeholder(tf.string, [], name='filename')
    image_file = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_file, channels=3, dct_method="INTEGER_ACCURATE")
    image = tf.to_float(image)
    self.image = image
    self.target_bbox_feed = tf.placeholder(dtype=tf.float32,
                                           shape=[4],
                                           name='target_bbox_feed')  # center's y, x, height, width

  def build_search_images(self):
    """Crop search images from the input image based on the last target position

    1. The input image is scaled such that the area of target&context takes up to (scale_factor * z_image_size) ^ 2
    2. Crop an image patch as large as x_image_size centered at the target center.
    3. If the cropped image region is beyond the boundary of the input image, mean values are padded.
    """
    model_config = self.model_config
    track_config = self.track_config

    size_z = model_config['z_image_size']
    size_x = track_config['x_image_size']
    context_amount = 0.5

    num_scales = track_config['num_scales']
    scales = np.arange(num_scales) - get_center(num_scales)
    assert np.sum(scales) == 0, 'scales should be symmetric'
    search_factors = [track_config['scale_step'] ** x for x in scales]

    frame_sz = tf.shape(self.image)
    target_yx = self.target_bbox_feed[0:2]
    target_size = self.target_bbox_feed[2:4]
    avg_chan = tf.reduce_mean(self.image, axis=(0, 1), name='avg_chan')

    # Compute base values
    base_z_size = target_size
    base_z_context_size = base_z_size + context_amount * tf.reduce_sum(base_z_size)
    base_s_z = tf.sqrt(tf.reduce_prod(base_z_context_size))  # Canonical size
    base_scale_z = tf.div(tf.to_float(size_z), base_s_z)
    d_search = (size_x - size_z) / 2.0
    base_pad = tf.div(d_search, base_scale_z)
    base_s_x = base_s_z + 2 * base_pad
    base_scale_x = tf.div(tf.to_float(size_x), base_s_x)

    boxes = []
    for factor in search_factors:
      s_x = factor * base_s_x
      frame_sz_1 = tf.to_float(frame_sz[0:2] - 1)
      topleft = tf.div(target_yx - get_center(s_x), frame_sz_1)
      bottomright = tf.div(target_yx + get_center(s_x), frame_sz_1)
      box = tf.concat([topleft, bottomright], axis=0)
      boxes.append(box)
    boxes = tf.stack(boxes)

    scale_xs = []
    for factor in search_factors:
      scale_x = base_scale_x / factor
      scale_xs.append(scale_x)
    self.scale_xs = tf.stack(scale_xs)

    # Note we use different padding values for each image
    # while the original implementation uses only the average value
    # of the first image for all images.
    image_minus_avg = tf.expand_dims(self.image - avg_chan, 0)
    image_cropped = tf.image.crop_and_resize(image_minus_avg, boxes,
                                             box_ind=tf.zeros((track_config['num_scales']), tf.int32),
                                             crop_size=[size_x, size_x])
    self.search_images = image_cropped + avg_chan

  def get_image_embedding(self, images, is_example, sa_siam_config, reuse=None):
    config = self.model_config['embed_config']
    arg_scope = sa_siam_arg_scope(config,
                                  trainable=config['train_embedding'],
                                  is_training=False)

    @functools.wraps(sa_siam)
    def embedding_fn(images, is_example, sa_siam_config, reuse=False):
      with slim.arg_scope(arg_scope):
        return sa_siam(images, is_example, sa_siam_config, reuse=reuse)

    embed, _ = embedding_fn(images=images, is_example=is_example, sa_siam_config=sa_siam_config, reuse=reuse)
    return embed

  def build_template(self):
    model_config = self.model_config
    track_config = self.track_config

    # Exemplar image lies at the center of the search image in the first frame
    exemplar_images = get_exemplar_images(self.search_images, [track_config['x_image_size'],
                                                               track_config['x_image_size']])
    templates = self.get_image_embedding(exemplar_images, is_example=True, sa_siam_config=self.model_config['sa_siam_config'])
    center_scale = int(get_center(track_config['num_scales']))
    center_template = tf.identity(templates[center_scale])
    templates = tf.stack([center_template for _ in range(track_config['num_scales'])])

    with tf.variable_scope('target_template'):
      # Store template in Variable such that we don't have to feed this template every time.
      with tf.variable_scope('State'):
        state = tf.get_variable('exemplar',
                                initializer=tf.zeros(templates.get_shape().as_list(), dtype=templates.dtype),
                                trainable=False)
        with tf.control_dependencies([templates]):
          self.init = tf.assign(state, templates, validate_shape=True)
        self.templates = state

  def build_detection(self):
    self.embeds = self.get_image_embedding(self.search_images, reuse=True, is_example=False, sa_siam_config=self.model_config['sa_siam_config'])
    with tf.variable_scope('detection'):
      def _get_mask_any(shape_mask, _u, _d, _l, _r):
        _mask = np.zeros(shape_mask, dtype='float32')
        _mask[_u:_d, _l:_r] = 1.0
        return _mask
      def _get_center_mask(shape_mask, _sz): # mask center a _sz x _sz patch
        _u = int((shape_mask[0] - _sz) / 2)
        _d = _u + _sz
        _l = int((shape_mask[1] - _sz) / 2)
        _r = _l + _sz
        return _get_mask_any(shape_mask, _u, _d, _l, _r)
      def _translation_match(x, z, mask_center=np.array([[1.0]], dtype='float32')):
        x = tf.expand_dims(x, 0)  # [batch, in_height, in_width, in_channels]
        z = tf.expand_dims(z, -1)  # [filter_height, filter_width, in_channels, out_channels]
        mask_center = tf.expand_dims(mask_center, -1)
        mask_center = tf.expand_dims(mask_center, -1)
        return tf.nn.conv2d(x, z * mask_center, strides=[1, 1, 1, 1], padding='VALID', name='translation_match')
      logging.info('Shape of templates: {}'.format(self.templates.shape))
      logging.info('Shape of embeds: {}'.format(self.embeds.shape))
      en_appearance = get(self.model_config['sa_siam_config'], 'en_appearance', False)
      en_semantic = get(self.model_config['sa_siam_config'], 'en_semantic', False)
      if en_appearance and en_semantic:
        c_appearance = get(self.model_config['sa_siam_config'], 'c_appearance', 0.3)
        out_scale = self.model_config['adjust_response_config']['scale'] 
        temp_appearance, temp_semantic = tf.split(self.templates, 2, 3)
        inst_appearance, inst_semantic = tf.split(self.embeds, 2, 3)
        bias_semantic = tf.get_variable('biases_semantic', [1],
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                              trainable=False)
        bias_appearance = tf.get_variable('biases_appearance', [1],
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                              trainable=False)
        sz_feat = shape_of(temp_appearance)[1:3] # [h,w]
        self.mask_all = {
          'keep_all': 1 - _get_center_mask(sz_feat, 0)
        }
        self.response_all = {}
        for k in sorted(self.mask_all.keys()):
          logging.info('Make match: {}'.format(k))
          match_k = lambda x: _translation_match(x[0], x[1], mask_center=self.mask_all[k])
          out_appearance_mask_k = tf.map_fn(match_k, (inst_appearance, temp_appearance), dtype=inst_appearance.dtype)
          out_semantic_mask_k  = tf.map_fn(match_k, (inst_semantic,  temp_semantic),  dtype=inst_semantic.dtype)

          out_appearance_mask_k = tf.squeeze(out_appearance_mask_k, [1,4])
          out_semantic_mask_k = tf.squeeze(out_semantic_mask_k, [1,4])

          response_appearance_mask_k = out_scale * out_appearance_mask_k
          response_semantic_mask_k  = out_scale * out_semantic_mask_k

          self.response_all[k] = (response_appearance_mask_k + bias_appearance) * c_appearance + (response_semantic_mask_k + bias_semantic) * (1-c_appearance) 
        response = self.response_all['keep_all']
      else:
        output = tf.map_fn(
          lambda x: _translation_match(x[0], x[1]),
          (self.embeds, self.templates), dtype=self.embeds.dtype)  # of shape [16, 1, 17, 17, 1]
        output = tf.squeeze(output, [1, 4])  # of shape e.g. [16, 17, 17]
        bias = tf.get_variable('biases', [1],
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                              trainable=False)
        response = (self.model_config['adjust_response_config']['scale'] * output + bias)
      self.response = response

  def build_upsample(self):
    """Upsample response to obtain finer target position"""
    with tf.variable_scope('upsample'):
      response = tf.expand_dims(self.response, 3)
      up_method = self.track_config['upsample_method']
      methods = {'bilinear': tf.image.ResizeMethod.BILINEAR,
                 'bicubic': tf.image.ResizeMethod.BICUBIC}
      up_method = methods[up_method]
      response_spatial_size = self.response.get_shape().as_list()[1:3]
      up_size = [s * self.track_config['upsample_factor'] for s in response_spatial_size]
      response_up = tf.image.resize_images(response,
                                           up_size,
                                           method=up_method,
                                           align_corners=True)
      response_up = tf.squeeze(response_up, [3])
      self.response_up = response_up

  def initialize(self, sess, input_feed):
    image_path, target_bbox = input_feed
    scale_xs, _ = sess.run([self.scale_xs, self.init],
                           feed_dict={'filename:0': image_path,
                                      "target_bbox_feed:0": target_bbox, })
    return scale_xs

  def inference_step(self, sess, input_feed):
    image_path, target_bbox = input_feed
    log_level = self.track_config['log_level']
    image_cropped_op = self.search_images if log_level > 0 else self.dumb_op
    image_cropped, scale_xs, response_output = sess.run(
      fetches=[image_cropped_op, self.scale_xs, self.response_up],
      feed_dict={
        "filename:0": image_path,
        "target_bbox_feed:0": target_bbox, })

    output = {
      'image_cropped': image_cropped,
      'scale_xs': scale_xs,
      'response': response_output}
    return output, None
