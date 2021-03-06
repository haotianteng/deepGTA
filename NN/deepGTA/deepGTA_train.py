#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 12:00:07 2017

@author: haotian.teng
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time,os

import deepGTA_input
import deepGTA

SNP_n = deepGTA_input.extract_SNP_n

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer(
      'max_steps',
      default_value=100000,
      docstring='Number of steps to run trainer.'
  )

tf.app.flags.DEFINE_string(
      'input_data_dir',
      default_value='/Users/haotian.teng/Documents/deepGTA/NN/Simulation/data/',
      docstring='Directory to put the input data.'
  )
tf.app.flags.DEFINE_string(
      'log_dir',
      default_value='/Users/haotian.teng/Documents/deepGTA/NN/logs/',
      docstring='Directory to put the log data.'
  )
tf.app.flags.DEFINE_boolean(
      'dummy_data',
      default_value=False,
      docstring='If true, uses fake data for unit testing.'
  )
tf.app.flags.DEFINE_string(
      'record_file',
      default_value='/Users/haotian.teng/Documents/deepGTA/NN/Test/R_squared_record_1.txt',
      docstring='Record file name to store the R_square score',
  )


def placeholder_input(batch_size):
    SNP  = tf.placeholder(dtype = tf.float32,name = 'x',shape = [batch_size,SNP_n])
    trait = tf.placeholder(dtype = tf.float32,name = 'y',shape = [batch_size,1])
    return SNP,trait

def fill_feed_dict(data_set, SNP_placeholder, trait_placeholder,shuffle = True):
    """fill the placeholder witht he data set gived
    Args:
        data_set: A Dataset class from deepGTA_input
        SNP_placeholder: A tf.placeholder with size [batch_size,SNP_n]
        trait_placeholder: A tf.placeholder with size [batch_size,1]
    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    SNP_batch,trait_batch = data_set.next_batch(FLAGS.batch_size,shuffle = shuffle)
    feed_dict = {SNP_placeholder:SNP_batch,trait_placeholder:trait_batch}
    return feed_dict

def run_once(sess,
            total_error,
            unexplained_error,
            SNP_placeholder,
            trait_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    R_squared: The Tensor that returns the R_squared of trait prediction.
    SNP_placeholder: The images placeholder.
    trait_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  total_e = 0  # Counts the number of correct predictions.
  residual = 0 #residual error
  steps_per_epoch = data_set.individual_n // FLAGS.batch_size
  num_samples = steps_per_epoch * FLAGS.batch_size
  for step in range(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               SNP_placeholder,
                               trait_placeholder,shuffle = False)
    total_e += sess.run(total_error, feed_dict=feed_dict)
    residual +=sess.run(unexplained_error,feed_dict = feed_dict)
  R_squared = 1 - float(residual) / float(total_e)
  print('  Num samples: %d  R_squared: %0.04f' %
        (num_samples, R_squared))
  return R_squared
  
def run_training():
   """Train deepGTA"""
   #read the data set
   data_sets = deepGTA_input.read_data_sets(FLAGS.input_data_dir, dummy_data = FLAGS.dummy_data)
    
   with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)   
       
    #placeholder for input data
    SNP_placeholder,trait_placeholder = placeholder_input(FLAGS.batch_size)
    
    #Build the graph to predict the trait
    trait_predict = deepGTA.inference(SNP_placeholder)
    
    #Add loss op into the graph
    loss = deepGTA.loss(trait_predict,trait_placeholder)
    
    #Add training op into graph
    train_step = deepGTA.training(loss,global_step)
    
    #Add Op to compare the predict trait with real trait and compute R2
    eval_loss,total_error,unexplained_error = deepGTA.evaluation(trait_predict,trait_placeholder)
    
    #Merge all the summary
    summary = tf.summary.merge_all()
    
    #Initialize
    init = tf.global_variables_initializer()
    
    #Create a savor for saving the model
    saver = tf.train.Saver(tf.global_variables())
    
    #Running
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(init)
    
    # Instantiate a SummaryWriter to output summaries and the Graph.
    if not os.path.exists(FLAGS.log_dir+'/summary'):
        os.makedirs(FLAGS.log_dir+'/summary')
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir+'/summary', sess.graph)
    
    for i in range(FLAGS.max_steps):
        start_time = time.time()
        feed_dict = fill_feed_dict(data_sets.train,SNP_placeholder,trait_placeholder,shuffle = False)
        _,loss_val,eval_loss_val = sess.run([train_step,loss,eval_loss],feed_dict = feed_dict)
        duration = time.time() - start_time              
        if i%100==0:
            print('Step %d: total_loss = %.2f  loss = %.2f (%.3f sec)' % (i, loss_val,eval_loss_val, duration))
            # Update the summary file.
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()
        if (i + 1) % 1000 == 0 or (i + 1) == FLAGS.max_steps:
            if not os.path.exists(FLAGS.log_dir+'/model'):
                os.makedirs(FLAGS.log_dir+'/model')
            checkpoint_file = os.path.join(FLAGS.log_dir+'/model', 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=i)
            # Evaluate against the training set.
            print('Training Data Eval:')
            R_squared_train = run_once(sess,
                    total_error,
                    unexplained_error,
                    SNP_placeholder,
                    trait_placeholder,
                    data_sets.train)
            # Evaluate against the validation set.
            print('Validation Data Eval:')
            R_squared_validation =run_once(sess,
                    total_error,
                    unexplained_error,
                    SNP_placeholder,
                    trait_placeholder,
                    data_sets.validation)
            # Evaluate against the test set.
            print('Test Data Eval:')
            R_squared_test =run_once(sess,
                    total_error,
                    unexplained_error,
                    SNP_placeholder,
                    trait_placeholder,
                    data_sets.test)
        
    return R_squared_train,R_squared_validation,R_squared_test
        
def main(_):
    R1,R2,R3 = run_training()
    R = [R1,R2,R3]
    if not os.path.exists(FLAGS.record_file):
        os.makedirs(FLAGS.record_file)
    with open(FLAGS.record_file,'w+') as f:
        f.write(','.join(str(r) for r in R)+'\n')
    
if __name__ == '__main__':
  tf.app.run()
    #FLAG_DICT = {input_data_dir,dummy_data,batch_size,learning_rate,l1_norm,max_step}
    