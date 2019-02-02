#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily          Huazhong University of Science and Technology
# Copyright © 2018 Anfeng He     Microsoft Research Asia. University of Science and Technology of China.
# Copyright (c) Microsoft. All rights reserved.
# 
# Distributed under terms of the MIT license.

r"""Support integration with OTB benchmark"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import sys
import time

import tensorflow as tf

sys.path.append(os.getcwd())

from configuration import LOG_DIR

# Code root absolute path
CODE_ROOT = './'

# Checkpoint for evaluation
CHECKPOINT = os.path.join(LOG_DIR, 'track_model_checkpoints', 'SA-Siam-Appearance', 'model.ckpt-{iter_ckpt}')

sys.path.insert(0, CODE_ROOT)

from utils.misc_utils import auto_select_gpu, load_cfgs
from inference import inference_wrapper
from inference.tracker import Tracker
from utils.infer_utils import Rectangle

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()
logging.getLogger().setLevel(logging.INFO)


def run_SA_Siam_Appearance(seq, rp, bSaveImage, epoch=30):
  iter_ckpt = epoch * 6650 - 1
  checkpoint_path = CHECKPOINT.format(iter_ckpt=iter_ckpt)
  logging.info('Evaluating {}...'.format(checkpoint_path))

  # Read configurations from json
  model_config, _, track_config = load_cfgs(checkpoint_path)

  track_config['log_level'] = 0  # Skip verbose logging for speed

  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(model_config, track_config, checkpoint_path)
  g.finalize()

  gpu_options = tf.GPUOptions(allow_growth=True)
  sess_config = tf.ConfigProto(gpu_options=gpu_options)

  with tf.Session(graph=g, config=sess_config) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    tracker = Tracker(model, model_config, track_config)

    tic = time.clock()
    frames = seq.s_frames
    init_rect = seq.init_rect
    x, y, width, height = init_rect  # OTB format
    init_bb = Rectangle(x - 1, y - 1, width, height)

    trajectory_py = tracker.track(sess, init_bb, frames)
    trajectory = [Rectangle(val.x + 1, val.y + 1, val.width, val.height) for val in
                  trajectory_py]  # x, y add one to match OTB format
    duration = time.clock() - tic

    result = dict()
    result['res'] = trajectory
    result['type'] = 'rect'
    result['fps'] = round(seq.len / duration, 3)
    return result
