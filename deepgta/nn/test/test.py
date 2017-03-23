#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:31:34 2017

@author: haotian.teng
"""
import tensorflow as tf 
#def test1():
#    with tf.variable_scope('test1'):
#        v1 = tf.Variable(initial_value = 2,name = 'v1',dtype = tf.float32)
#        v2 = tf.get_variable('v2',[1,2],initializer = tf.random_normal_initializer(),dtype = tf.float32)
#        v3 = tf.placeholder(dtype = tf.float32,name = 'v3',shape = [2])
#        v4 = v1+v2+v3
#    return v1,v2,v3,v4
#sess = tf.Session()
#init = tf.initialize_all_variables()
#sess.run(init)
#v4_val = sess.run(v4,feed_dict = {v3:[2,2]})
#saver = tf.train.Saver(tf.global_variables())
#sess.run(tf.global_variables_initializer())
#saver.save(sess,'my_model')
#print(v1.name)

with tf.Graph().as_default() as g_1:
  x = tf.placeholder(tf.float32, name="input")
  y = x+1
  # NOTE: using identity to get a known name for the output tensor.
  output = tf.identity(y, name="output")

gdef_1 = g_1.as_graph_def()

with tf.Graph().as_default() as g_2:  # NOTE: g_2 not g_1       
  x = tf.placeholder(tf.float32, name="input")
  z = x*2
  output = tf.identity(z, name="output")

gdef_2 = g_2.as_graph_def()

with tf.Graph().as_default() as g_combined:
  x = tf.placeholder(tf.float32, name="x_hat")

  # Import gdef_1, which performs f(x).
  # "input:0" and "output:0" are the names of tensors in gdef_1.
  y, = tf.import_graph_def(gdef_1, input_map={"input:0": x},
                           return_elements=["output:0"])

  # Import gdef_2, which performs g(y)
  z, = tf.import_graph_def(gdef_2, input_map={"input:0": y},
                           return_elements=["output:0"])