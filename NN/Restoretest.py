#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 18:02:10 2017

@author: haotian.teng
"""

import tensorflow as tf 
def test1():
    with tf.variable_scope('test1'):
        v1 = tf.Variable(2,name = 'v1')
        v2 = tf.get_variable('v2',[1,2],initializer = tf.constant_initializer(0.0))
        v3 = tf.placeholder(dtype = tf.float32,name = 'v3',shape = [2])
    return v1,v2,v3
#v1,v2,v3 = test1()
with tf.variable_scope('test1'):
    v1 = tf.Variable(2,name = 'v1')
    v2 = tf.get_variable('v2',[1,2],initializer = tf.constant_initializer(0.0))
    v3 = tf.placeholder(dtype = tf.float32,name = 'v3',shape = [2])
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess,'/Users/haotian.teng/Documents/ComplexTraitANN/my_model')

print(v1.name)