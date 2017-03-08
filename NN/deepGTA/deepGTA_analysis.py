#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:08:06 2017

@author: haotian.teng
"""
import tensorflow as tf,numpy as np
import deepGTA
import matplotlib.pyplot as plt

def placeholder_input(batch_size):
    SNP  = tf.placeholder(dtype = tf.float32,name = 'x',shape = [batch_size,deepGTA.SNP_n])
    trait = tf.placeholder(dtype = tf.float32,name = 'y',shape = [batch_size,1])
    return SNP,trait

def load_model(save_file,step_index):
    file_path = save_file + '-' + str(step_index)
    SNP_placeholder,trait_placeholder = placeholder_input(1)
    
    #Build the graph to predict the trait
    deepGTA.inference(SNP_placeholder)
    ckpt = tf.train.import_meta_graph(file_path+'.meta')
    return ckpt

step_index = 9999
file_path = "/Users/haotian.teng/Documents/deepGTA/NN/logs/model.ckpt"
ckpt = load_model(file_path,step_index)
sess = tf.Session()
with tf.variable_scope("hidden1",reuse = True):
    w1 = tf.get_variable("weights")
    b1 = tf.get_variable("bias")
with tf.variable_scope("hidden2",reuse = True):
    w2 = tf.get_variable("weights")
    b2 = tf.get_variable("bias")
with tf.variable_scope("softmax_linear",reuse = True):
    w = tf.get_variable("weights")
    b = tf.get_variable("bias")
init = tf.global_variables_initializer()
sess.run(init)
ckpt.restore(sess,file_path+'-'+str(step_index))
w1_value = sess.run(w1)
w2_value = sess.run(w2)
w_value = sess.run(w)
with open("/Users/haotian.teng/Documents/deepGTA/NN/Simulation/dataeffect_index.dat",'r') as f:
    for line in f:
        if line.startswith('#Index'):
            effect_index = next(f).split(',')
            effect_index = [int(x) for x in effect_index]
        if line.startswith('#Effect_size'):
            effect_size = next(f).split(',')
            effect_size = [float(x) for x in effect_size]
w1_mean = np.mean(w1_value)
w1_std = np.std(w1_value)
w1_cap = w1_value>(w1_mean+2*w1_std)
plt.imshow(w1_cap)

w2_mean = np.mean(w2_value)
w2_std = np.std(w2_value)
w2_cap = w2_value>(w2_mean+2*w2_std)
plt.imshow(w2_cap)