#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 16:25:09 2017

@author: haotian.teng
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import deepGTA_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
      'model_restore',
      default_value = "/Users/haotian.teng/Documents/deepGTA/NN/logs/model/model.ckpt",
      docstring ='Reload the model to train instead of a new model.')

def load_graph(FLAGS.model_restore,global_step = i):
    sess = tf.Session()
    ckpt = tf.train.import_meta_graph(FLAGS.model_restore+'-'+str(global_step))
    