# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:31:16 2017

@author: Heavens
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

### Hyper parameter
SNP_n = 1000

def inference(SNP):
    """Build the deepGTA model here:
    Args:
        SNP: tf.placeholder, from the deepGTA_input.placeholder_input()
    Returns:
        trait_predict: Predicted Trait, 1D tensor of [batch_size]
    """        
    with tf.variable_scope("FNN"):
        #First Layer
        w1 = tf.get_variable("w1",shape = [SNP_n,1],dtype = tf.float32, initializer = tf.constant_initializer(0.0))
        b1 = tf.get_variable("b1",shape = 1, dtype = tf.float32, initializer = tf.constant_initializer(0.0))
        #First Layer end
        trait_predict = tf.matmul(SNP,w1)+b1
    return trait_predict
def loss(trait_predict,trait,l1_coeff):
    """Calculate the loss:
    Args:
        trait_predict: Predicted trait value from inference()
        trait: True trait value
        l1_coeff: the l1 regularization strength
    Returns:
        loss: Loss with 1st norm regularization of weight
            
    """
    #Get the weight
    with tf.variable_scope("FNN",reuse = True):
        w1 = tf.get_variable("w1")
        tf.summary.histogram('w1',w1)
    #Calculate the loss with 1st Norm Regularization of weight
    loss = tf.reduce_mean(tf.square(trait_predict-trait)) + l1_coeff*tf.reduce_sum(tf.abs(w1))
    return loss
def training(loss,learning_rate):
    """Training the model:
        Args:
            loss:loss from the loss()
        Returns:
            train:tensorflow operation for training
    """
    
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    return train
    
def evaluation(trait_predict,trait):
    """Evaluate the perfermance:
        Args:
            trait_predict: Predicted trait value from inference()
            trait: True trait value
        Returns:
            eva_loss:Tensorflow tensor record the loss.
            total_error:Tensor record the total variation in the quantitive trait.
            unedplained_error: Tensor record the Unexplained variation.
    """
    eva_loss = tf.reduce_mean(tf.square(trait_predict-trait))
    total_error = tf.reduce_sum(tf.square(trait-tf.reduce_mean(trait)))
    unexplained_error = tf.reduce_sum(tf.square(trait-trait_predict))
    return eva_loss,total_error,unexplained_error