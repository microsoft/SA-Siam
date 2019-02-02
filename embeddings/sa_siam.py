#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily          Huazhong University of Science and Technology
# Copyright © 2018 Anfeng He     Microsoft Research Asia. University of Science and Technology of China.
# Copyright (c) Microsoft. All rights reserved.
#
# Distributed under terms of the MIT license.

"""Contains definitions of the network in [1][2].

  [1] Bertinetto, L., et al. (2016).
      "Fully-Convolutional Siamese Networks for Object Tracking."
      arXiv preprint arXiv:1606.09549.
  [2] Anfeng He, et al. (2018).
      "A Twofold Siamese Network for Real-Time Object Tracking."
      arXiv preprint arXiv:1802.08817.

Typical use:

   import sa_siam
   with slim.arg_scope(sa_siam.sa_siam_arg_scope()):
      net, end_points = sa_siam.sa_siam(inputs, is_training=False)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging 
logging.getLogger().setLevel(logging.INFO)

import tensorflow as tf

from utils.misc_utils import get, shape_of, same_hw
from functools import reduce 

slim = tf.contrib.slim


def sa_siam_arg_scope(embed_config,
                      trainable=True,
                      is_training=False):
  """Defines the default arg scope.

  Args:
    embed_config: A dictionary which contains configurations for the embedding function.
    trainable: If the weights in the embedding function is trainable.
    is_training: If the embedding function is built for training.

  Returns:
    An `arg_scope` to use for the SA-Siam models.
  """
  # Only consider the model to be in training mode if it's trainable.
  # This is vital for batch_norm since moving_mean and moving_variance
  # will get updated even if not trainable.
  is_model_training = trainable and is_training

  if get(embed_config, 'use_bn', True):
    batch_norm_scale = get(embed_config, 'bn_scale', True)
    batch_norm_decay = 1 - get(embed_config, 'bn_momentum', 3e-4)
    batch_norm_epsilon = get(embed_config, 'bn_epsilon', 1e-6)
    batch_norm_params = {
      "scale": batch_norm_scale,
      # Decay for the moving averages.
      "decay": batch_norm_decay,
      # Epsilon to prevent 0s in variance.
      "epsilon": batch_norm_epsilon,
      "trainable": trainable,
      "is_training": is_model_training,
      # Collection containing the moving mean and moving variance.
      "variables_collections": {
        "beta": None,
        "gamma": None,
        "moving_mean": ["moving_vars"],
        "moving_variance": ["moving_vars"],
      },
      'updates_collections': None,  # Ensure that updates are done within a frame
    }
    normalizer_fn = slim.batch_norm
  else:
    batch_norm_params = {}
    normalizer_fn = None

  weight_decay = get(embed_config, 'weight_decay', 5e-4)
  if trainable:
    weights_regularizer = slim.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  init_method = get(embed_config, 'init_method', None)
  if is_model_training:
    logging.info('embedding init method -- {}'.format(init_method))
  if init_method == 'kaiming_normal':
    # The same setting as siamese-fc
    initializer = slim.variance_scaling_initializer(factor=2.0, mode='FAN_OUT', uniform=False)
  else:
    initializer = slim.xavier_initializer()

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=weights_regularizer,
      weights_initializer=initializer,
      padding='VALID',
      trainable=trainable,
      activation_fn=tf.nn.relu,
      normalizer_fn=normalizer_fn,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.batch_norm], is_training=is_model_training) as arg_sc:
          return arg_sc
def appearance_net(layer_in):
  logging.info('Building Appearence branch of SA-Siam')
  layers_all = []
  layer_cur = slim.conv2d(layer_in, 96, [11, 11], 2, scope='conv1')
  layer_cur = slim.max_pool2d(layer_cur, [3, 3], 2, scope='pool1')
  layers_all.append(layer_cur)
  with tf.variable_scope('conv2'):
    b1, b2 = tf.split(layer_cur, 2, 3)
    b1 = slim.conv2d(b1, 128, [5, 5], scope='b1')
    # The original implementation has bias terms for all convolution, but
    # it actually isn't necessary if the convolution layer is followed by a batch
    # normalization layer since batch norm will subtract the mean.
    b2 = slim.conv2d(b2, 128, [5, 5], scope='b2')
    layer_cur = tf.concat([b1, b2], 3)
  layer_cur = slim.max_pool2d(layer_cur, [3, 3], 2, scope='pool2')
  layers_all.append(layer_cur)
  layer_cur = slim.conv2d(layer_cur, 384, [3, 3], 1, scope='conv3')
  layers_all.append(layer_cur)
  with tf.variable_scope('conv4'):
    b1, b2 = tf.split(layer_cur, 2, 3)
    b1 = slim.conv2d(b1, 192, [3, 3], 1, scope='b1')
    b2 = slim.conv2d(b2, 192, [3, 3], 1, scope='b2')
    layer_cur = tf.concat([b1, b2], 3)
    layers_all.append(layer_cur)
  # Conv 5 with only convolution
  with tf.variable_scope('conv5'):
    with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
      b1, b2 = tf.split(layer_cur, 2, 3)
      b1 = slim.conv2d(b1, 128, [3, 3], 1, scope='b1')
      b2 = slim.conv2d(b2, 128, [3, 3], 1, scope='b2')
    layer_cur = tf.concat([b1, b2], 3)
    layers_all.append(layer_cur)
  return layer_cur, layers_all

def semantic_net(layer_in):
  logging.info('Building Semantic branch of SA-Siam..')
  layers_all = []
  with slim.arg_scope([slim.conv2d], normalizer_fn=None, trainable=False, normalizer_params=False):
    layer_cur = layer_in - [123.0,117.0,104.0] # RGB sub mean
    layer_cur = tf.reverse(layer_cur,[3]) # convert img to BGR
    layer_cur = slim.conv2d(layer_cur, 96, [11, 11], 2, scope='conv1')
    layer_cur = slim.max_pool2d(layer_cur, [3, 3], 2, scope='pool1')
    layer_cur = tf.nn.local_response_normalization(layer_cur,depth_radius=2,alpha=2e-5,beta=0.75,bias=1.0,name='norm1')
    layers_all.append(layer_cur)
    with tf.variable_scope('conv2'):
      b1, b2 = tf.split(layer_cur, 2, 3)
      b1 = slim.conv2d(b1, 128, [5, 5], scope='b1')
      b2 = slim.conv2d(b2, 128, [5, 5], scope='b2')
      layer_cur = tf.concat([b1, b2], 3)
    layer_cur = slim.max_pool2d(layer_cur, [3, 3], 2, scope='pool2')
    layer_cur = tf.nn.local_response_normalization(layer_cur,depth_radius=2,alpha=2e-5,beta=0.75,bias=1.0,name='norm2')
    layers_all.append(layer_cur)
    layer_cur = slim.conv2d(layer_cur, 384, [3, 3], 1, scope='conv3')
    layers_all.append(layer_cur)
    with tf.variable_scope('conv4'):
      b1, b2 = tf.split(layer_cur, 2, 3)
      b1 = slim.conv2d(b1, 192, [3, 3], 1, scope='b1')
      b2 = slim.conv2d(b2, 192, [3, 3], 1, scope='b2')
      layer_cur = tf.concat([b1, b2], 3)
      layers_all.append(layer_cur)
    # Conv 5 with only convolution
    with tf.variable_scope('conv5'):
      with slim.arg_scope([slim.conv2d],activation_fn=tf.nn.relu):
        b1, b2 = tf.split(layer_cur, 2, 3)
        b1 = slim.conv2d(b1, 128, [3, 3], 1, scope='b1')
        b2 = slim.conv2d(b2, 128, [3, 3], 1, scope='b2')
    layer_cur = tf.concat([b1, b2], 3)
    layers_all.append(layer_cur)
  return layer_cur, layers_all
def combine_sa_net(a_net, s_net):
  all_feat = a_net + s_net
  assert(all(list(map(same_hw, all_feat))))
  max_feat_size = max(list(map(lambda a: shape_of(a)[1], all_feat)))
  logging.info('Max_feat_size={}'.format(max_feat_size))
  def pad_feat(feat):
    if max_feat_size is None and shape_of(feat)[1] is None:
      return feat
    pad_size = max_feat_size - shape_of(feat)[1]
    pad_l = pad_size // 2
    pad_r = pad_size - pad_l
    return tf.pad(feat,[[0,0],[pad_l,pad_r],[pad_l,pad_r],[0,0]])
  all_feat = list(map(pad_feat, all_feat))
  return tf.concat(all_feat, axis=3)

def sa_siam(inputs,
            is_example,
            sa_siam_config={},
            reuse=None,
            scope='sa_siam'):
  en_appearance = get(sa_siam_config, 'en_appearance', False)
  en_semantic = get(sa_siam_config, 'en_semantic', False)
  n_out = get(sa_siam_config, 'n_out', 256)
  all_combine_layers_appearance = get(sa_siam_config, 'all_combine_layers_appearance', {'conv5':1.0})
  all_combine_layers_semantic = get(sa_siam_config, 'all_combine_layers_semantic', {'conv5':1.0, 'conv4':0.1})
  sz_conv5_z = get(sa_siam_config, 'sz_conv5_z', 6)
  en_semantic_att = get(sa_siam_config, 'en_semantic_att', True)

  with tf.variable_scope(scope, 'sa_siam', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      def proc_raw_all_feat(feat, is_appearance, n_out_cur, all_combine_layers):
        res = []
        max_feat_size = 0
        for l in range(1,6):
          for k in all_combine_layers.keys():
            if k.find(str(l)) != -1:
              if shape_of(feat[l-1])[3] is None:
                res.append(feat[l-1])
                break
              if l == 5 and is_appearance and abs(n_out_cur - shape_of(feat[l-1])[3]) < 0.1:
                res.append(feat[l-1])
              else:
                if not is_appearance:
                  feat[l-1] *= all_combine_layers[k] # Multiple scale for convergence during training
                with slim.arg_scope([slim.conv2d],activation_fn=None, normalizer_fn=None):
                  c1x1 = slim.conv2d(feat[l-1], n_out_cur, [1,1], 1, scope='c1x1_' + k)
                res.append(c1x1)
              logging.info('Keep {} .. is_appearance={} shape={}'.format(k,is_appearance,shape_of(res[-1])))
        return res
      def re_weight_crop(feat, all_combine_layers, only_crop=False):
        feat_shape = list(map(shape_of, feat))
        res = []
        for l in range(1,6): # proc layers from 1 to 5 in order
          for k in all_combine_layers.keys(): # find the corresponding layer in all layers
            if k.find(str(l)) != -1:
              logging.info('For layer {} ...'.format(k))
              cur_ly_idx = l - 1
              if feat_shape[cur_ly_idx][2] is None and feat_shape[4][2] is None:
                res.append(feat[cur_ly_idx])
                break
              pad_val = feat_shape[cur_ly_idx][2] - feat_shape[4][2]
              sz_conv5_z_cur = pad_val + sz_conv5_z
              sz_conv5_x_cur = feat_shape[cur_ly_idx][2]
              n_left = int((sz_conv5_x_cur - sz_conv5_z_cur) / 2 + 0.5)
              div_left_st = [0, n_left, n_left + sz_conv5_z_cur, sz_conv5_x_cur]
              logging.info('.. Crop as {}'.format(div_left_st)) # crop 9 patchs and max pool each patch
              if not only_crop:
                all_max = []
                for j in [0,1,2]:
                  for i in [0,1,2]:
                    l_crop = div_left_st[i]
                    r_crop = div_left_st[i + 1]
                    u_crop = div_left_st[j]
                    d_crop = div_left_st[j+1]
                    max_patch = tf.reduce_max(feat[cur_ly_idx][:, u_crop:d_crop, l_crop:r_crop, :], axis=[1, 2]) #shape = [n, c]
                    all_max.append(max_patch)
                max_map = tf.stack(all_max, axis=2) #shape = [n, c, 9]
                logging.info('.. Max_map.shape = {}'.format(max_map.shape))
                max_map = slim.fully_connected(max_map, 9, scope='att_fc1_' + k) # fully_connected layer will only applied to the last dim
                logging.info('.. Max_map_fc1.shape = {}'.format(max_map.shape))
                max_map = slim.fully_connected(max_map, 1, scope='att_fc2_' + k, activation_fn=None, normalizer_fn=None,)
                logging.info('.. Max_map_fc2.shape = {}'.format(max_map.shape)) # shape = [n, c, 1]
                att_map = tf.reshape(max_map, [-1, 1, 1, feat_shape[cur_ly_idx][3]])
                logging.info('.. att_map.shape = {}'.format(att_map.shape))
                att_map = tf.sigmoid(att_map) + 0.5 # important bias for avoiding loss too much
                feat[cur_ly_idx] = att_map * feat[cur_ly_idx]
              feat[cur_ly_idx] = feat[cur_ly_idx][:, div_left_st[1]:div_left_st[2], div_left_st[1]:div_left_st[2], :] # crop center feat
              res.append(feat[cur_ly_idx])
              break
          else:
            res.append(None)
        return res
      layer_cur = inputs
      if en_appearance:
        n_out_appearance = n_out / len(all_combine_layers_appearance.keys())
        with tf.variable_scope('appearance_net'):
          _, feat_appearance_all = appearance_net(layer_cur)
          if is_example:
            feat_appearance_all = re_weight_crop(feat_appearance_all, all_combine_layers_appearance, only_crop=True)
          net_appearance = proc_raw_all_feat(feat_appearance_all, is_appearance=True, n_out_cur=n_out_appearance, all_combine_layers=all_combine_layers_appearance)
      if en_semantic:
        n_out_semantic = n_out / len(all_combine_layers_semantic.keys())
        with tf.variable_scope('semantic_net'):
          _, feat_semantic_all = semantic_net(layer_cur)
          if is_example:
            feat_semantic_all = re_weight_crop(feat_semantic_all, all_combine_layers_semantic, only_crop=not en_semantic_att)
          net_semantic = proc_raw_all_feat(feat_semantic_all, is_appearance=False, n_out_cur=n_out_semantic, all_combine_layers=all_combine_layers_semantic)
      if en_appearance and en_semantic:
        layer_cur = combine_sa_net(net_appearance, net_semantic)
      elif en_appearance:layer_cur = combine_sa_net(net_appearance, [])
      elif en_semantic:layer_cur = combine_sa_net(net_semantic, [])
      else: raise ValueError('Semantic or Appearance must enable one branch!')
      # Convert end_points_collection into a dictionary of end_points.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return layer_cur, end_points

sa_siam.stride = 8
