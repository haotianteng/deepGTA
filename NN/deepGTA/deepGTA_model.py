# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:31:16 2017

@author: Heavens
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.platform import flags
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
### Hyper parameter
#Graph
SNP_n = 1000
Batch_size = 500

#Train
Epoch = 20000
TrainProp = 0.8
TestProp = 1 - TrainProp
train_step = 0.001

### Read the data
geno_f = open('train_geno.dat','r')
pheno_f = open('train_pheno.dat','r')
index_f = open('effect_index.dat','r')
geno_data = list()
pheno_data = list()
for line in geno_f:
    geno_data.append([float(x) for x in line.split(',')])
for line in pheno_f:
    pheno_data.append([float(x) for x in line.split(',')])
geno_data = np.asarray(geno_data)
pheno_data = np.asarray([[x] for x in pheno_data[0]])

Sample_n = len(geno_data)
train_n = int(Sample_n*TrainProp)
geno_train = geno_data[:train_n]
pheno_train = pheno_data[:train_n]
geno_test = geno_data[train_n:]
pheno_test = pheno_data[train_n:]
assert geno_train.shape[1]==SNP_n,"Input SNP number is %d, expect %d."%(geno_train.shape[1],SNP_n)

class DeepGTA_Model:
    x = None
    y = None
    def __init__(self):        
        with tf.variable_scope("FNN"):
            x  = tf.placeholder(dtype = tf.float32,name = 'x',shape = [None,SNP_n])
            y = tf.placeholder(dtype = tf.float32,name = 'y',shape = [None,1])
            #First Layer
            w1 = tf.get_variable("w1",shape = [SNP_n,1],dtype = tf.float32, initializer = tf.constant_initializer(0.0))
            b1 = tf.get_variable("b1",shape = 1, dtype = tf.float32, initializer = tf.constant_initializer(0.0))
            #First Layer end
            y_hat = tf.matmul(x,w1)+b1
            loss = tf.reduce_mean(tf.square(y_hat-y)) + 0.01*tf.reduce_sum(tf.abs(w1))
            
            #Evaluation
            eva_loss = tf.reduce_mean(tf.square(y_hat-y))
            total_error = tf.reduce_sum(tf.square(y-tf.reduce_mean(y)))
            unexplained_error = tf.reduce_sum(tf.square(y-y_hat))
            R_square = 1-tf.div(unexplained_error,total_error)
            
            optimizer = tf.train.GradientDescentOptimizer(train_step)
            train = optimizer.minimize(loss)
            init = tf.global_variables_initializer()
            self.x = x
            self.y = y
        
    def training(self):
        sess = tf.Session()
        sess.run(init)
        max_batch = train_n // Batch_size
        for i in range(Epoch):
            batch_index = i%max_batch
            x_data = geno_train[batch_index*500:(batch_index+1)*500]
            y_data = pheno_train[batch_index*500:(batch_index+1)*500]
            [batch_loss,_] = sess.run([loss,train],feed_dict = {x:x_data,y:y_data})
            if i%100==0:
                print("Traning set batch_loss %s"%(batch_loss))
                eva_loss_val = sess.run(eva_loss,feed_dict = {x:geno_test,y:pheno_test})
                print("Test set loss %f"%(eva_loss_val))
        
        
    def evaluation(self):
        w1_val = sess.run(w1)
        for line in index_f:
            if line.startswith('#Index'):
                Index_line = next(index_f)
                Index = [int(x) for x in Index_line.split(',')]
            if line.startswith('#Effect_size'):
                Effect_line = next(index_f)
                Effect_size = [float(x) for x in Effect_line.split(',')]
        max_noneffect = 0
        noeffect = np.delete(w1_val,Index)
        np.max(np.abs(noeffect))
        max_noneffect = [x]
        plt.plot(w1_val[Index],Effect_size,'ro')