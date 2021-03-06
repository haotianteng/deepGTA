#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 18:02:10 2017

@author: haotian.teng
"""

import tensorflow as tf 
def test1():
    with tf.variable_scope('test1'):
        v1 = tf.Variable(initial_value = 2,name = 'v1',dtype = tf.float32)
        v2 = tf.get_variable('v2',[1,2],initializer = tf.constant_initializer(0.0),dtype = tf.float32)
        v3 = tf.placeholder(dtype = tf.float32,name = 'v3',shape = [2])
        v4 = v1+v2
    return v1,v2,v3,v4
v1,v2,v3,v4 = test1()
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess,'/Users/haotian.teng/Documents/deepGTA/NN/my_model')

print(v1.name)