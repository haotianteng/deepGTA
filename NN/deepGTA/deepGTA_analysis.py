#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:08:06 2017

@author: haotian.teng
"""
import tensorflow as tf
import deepGTA

def placeholder_input(batch_size):
    SNP  = tf.placeholder(dtype = tf.float32,name = 'x',shape = [batch_size,deepGTA.SNP_n])
    trait = tf.placeholder(dtype = tf.float32,name = 'y',shape = [batch_size,1])
    return SNP,trait

def load_model(save_file,step_index):
    file_path = save_file + '-' + str(step_index)
    SNP_placeholder,trait_placeholder = placeholder_input(1)
    
    #Build the graph to predict the trait
    deepGTA.inference(SNP_placeholder)
    
    sess = tf.Session()
    ckpt = tf.train.import_meta_graph(file_path+'.meta')
    ckpt.restore(sess,file_path)
    return sess

def main():
    step_index = 27999
    file_path = "/Users/haotian.teng/Documents/deepGTA/NN/logs/model.ckpt"
    sess = load_model(file_path,step_index)
    with tf.variable_scope("hidden1",reuse = True):
        w1 = tf.get_variable("weights")
    sess.run(w1)

main()