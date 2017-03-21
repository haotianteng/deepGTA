# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 14:31:16 2017

@author: Heavens
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import deepGTA_input
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer(
      'batch_size',
      default_value=200,
      docstring='Batch size.  Must divide evenly into the dataset sizes.'
  )

tf.app.flags.DEFINE_float(
      'learning_rate',
      default_value=0.01,
      docstring='Initial learning rate.'
  )
tf.app.flags.DEFINE_float(
      'weight_decay',
      default_value=0.0001,
      docstring='Weight decay strength of the regularization term'
  )
tf.app.flags.DEFINE_string(
       'decay_method',
       default_value = 'l1',
       docstring = 'Method used to calculate regularization of the weight, typically l1,l2.'
       )
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = deepGTA_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_VALIDATION = deepGTA_input.NUM_EXAMPLES_PER_EPOCH_FOR_VALIDATION
NUM_EXAMPLES_FOR_TEST = deepGTA_input.NUM_EXAMPLES_FOR_TEST
SNP_n = deepGTA_input.extract_SNP_n

NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
KEEP_PROB = [1.0,1.0]
def _activation_summary(x):
  """Helper to create summaries for activations.
  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.
  Args:
    x: Tensor
  Returns:
    nothing
  """
  # This helps the clarity of presentation on tensorboard.
  tensor_name = x.op.name
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
 
def _variable_with_regularization(name,shape,stddev,wd,method = "l2"):
    """Create a variable with regularization which is initailized with truncated
    normal initializer.
    Args:
        name: name of the variable
        shape: shape of the variable, expect a list
        stddev: standard deviation of initailization distribution
        wd: weight decat cofficient.
        method: regularization method used, typically l1,l2
        
    Returns:
        Variable of tensor
    """
    var = tf.get_variable(name = name, shape = shape,dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev = stddev))
    if wd:
        if method == "l1":
            regularization = tf.multiply(tf.reduce_sum(tf.abs(var)),wd,name = "weight_loss")
        if method == "l2":
            regularization = tf.multiply(tf.nn.l2_loss(var),wd,name = "weight_loss")
        tf.add_to_collection('losses',regularization)
    return var
    
def inference(SNP):
    """Build the deepGTA model here:
    Args:
        SNP: tf.placeholder, from the deepGTA_input.placeholder_input(),shape = [batch_size,SNP_n]
    Returns:
        trait_predict: Predicted Trait, 1D tensor of [batch_size]
    """
    
    hidden1_units = 1
    hidden2_units = 1
    weight_decay = FLAGS.weight_decay
    decay_method = FLAGS.decay_method
    with tf.variable_scope("hidden1"):
        #First Layer variable
        w = _variable_with_regularization("weights",[SNP_n,hidden1_units],0.04,weight_decay,decay_method)
        b = tf.get_variable("bias",shape = [hidden1_units], dtype = tf.float32, initializer = tf.constant_initializer(0.01))
        #Operation
        z = tf.nn.bias_add(tf.matmul(SNP,w),b)
        #z1 will be a [batch_size,hidden1_units] tensor
        hidden1 = tf.nn.relu(z)
        #Dropout to improve the training result
        keep_prob1 = tf.constant(KEEP_PROB[0],name = "keep_prob_1")
        hidden1_dropout = tf.nn.dropout(hidden1,keep_prob1)
        _activation_summary(hidden1)
    
    with tf.variable_scope("hidden2"):
        #Second Layer
        w = _variable_with_regularization("weights",[hidden1_units,hidden2_units],0.04,weight_decay,decay_method)
        b = tf.get_variable("bias",shape = [hidden2_units], dtype = tf.float32, initializer = tf.constant_initializer(0.01))
        #Operation
        z = tf.nn.bias_add(tf.matmul(hidden1_dropout,w),b)
        #z2 will be a [batch_size,hidden2_units] tensor
        hidden2 = tf.nn.relu(z)
        #Dropout to improve the training result
        keep_prob2 = tf.constant(KEEP_PROB[1],name = "keep_prob_2")
        hidden2_dropout = tf.nn.dropout(hidden2,keep_prob2)
        _activation_summary(hidden2)
    
    with tf.variable_scope("softmax_linear"):
        w = _variable_with_regularization("weights",[hidden2_units,1],0.04,weight_decay,decay_method)
        b = tf.get_variable("bias",shape = 1,dtype = tf.float32,initializer = tf.constant_initializer(0.1))
        
        trait_predict = tf.nn.bias_add(tf.matmul(hidden2_dropout,w),b)
    
    return trait_predict
def inference_linear(SNP):
    """Build a linear model here:
        Args:
        SNP: tf.placeholder, from the deepGTA_input.placeholder_input(),shape = [batch_size,SNP_n]
    Returns:
        trait_predict: Predicted Trait, 1D tensor of [batch_size]
    """
    with tf.variable_scope("softmax_linear"):
        w = _variable_with_regularization("weights",[SNP_n,1],0.01,FLAGS.weight_decay,FLAGS.decay_method)
        b = tf.get_variable("bias",shape = 1,dtype = tf.float32,initializer = tf.constant_initializer(0.1))
        trait_predict = tf.nn.bias_add(tf.matmul(SNP,w),b)
    return trait_predict
    
    
def loss(trait_predict,trait):
    """Calculate the loss:
    Args:
        trait_predict: Predicted trait value from inference()
        trait: True trait value
        l1_coeff: the l1 regularization strength
    Returns:
        loss: Loss with 1st norm regularization of weight
            
    """

    #Calculate the loss 
    loss = tf.reduce_mean(tf.square(trait_predict-trait))
    tf.add_to_collection('losses',loss)
    return tf.add_n(tf.get_collection('losses'),name = 'total_loss')

def training(loss,global_step):
    """Training the model:
        Args:
            loss:loss from the loss()
        Returns:
            train:tensorflow operation for training
    """
    
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
    tf.summary.scalar('learning_rate', lr)
    optimizer = tf.train.GradientDescentOptimizer(lr)
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