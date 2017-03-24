#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 12:00:07 2017

@author: haotian.teng
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf,numpy as np
import time,os
import deepgta_input
import deepgta_model

default_log_dir = "/Users/haotian.teng/Documents/deepgta/deepgta/nn/logs"
default_record_dir = "/Users/haotian.teng/Documents/deepgta/deepgta/nn/logs/record"

SNP_n = deepgta_input.extract_SNP_n

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer(
      'max_steps',
      default_value=3000,
      docstring='Number of steps to run trainer.'
  )

tf.app.flags.DEFINE_string(
      'geno_file',
      default_value=None,
      docstring='Path to the input genome data, there should be three files including .bed .bim .fam in the directory with same file name.'
  )
tf.app.flags.DEFINE_string(
      'pheno_file',
      default_value=None,
      docstring = 'Path to the input trait data, first row of the file should include a name list of the traits.')
tf.app.flags.DEFINE_string(
      'trait_name',
      default_value = None,
      docstring = 'Trait name.')
tf.app.flags.DEFINE_string(
      'log_dir',
      default_value=default_log_dir,
      docstring='Directory to put the log data.'
  )
tf.app.flags.DEFINE_boolean(
      'dummy_data',
      default_value=False,
      docstring='If true, uses fake data for unit testing.'
  )
tf.app.flags.DEFINE_string(
      'record_file',
      default_value=default_record_dir+'/R_squared_record_1.txt',
      docstring='Record file name to store the R_square score',
  )


def placeholder_input(batch_size):
    SNP  = tf.placeholder(dtype = tf.float32,name = 'x',shape = [batch_size,SNP_n])
    trait = tf.placeholder(dtype = tf.float32,name = 'y',shape = [batch_size,1])
    return SNP,trait

def fill_feed_dict(data_set, SNP_placeholder, trait_placeholder,shuffle = True):
    """fill the placeholder witht he data set gived
    Args:
        data_set: A Dataset class from deepgta_input
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
   """Train deepgta_model"""
   #read the data set
   data_sets = deepgta_input.read_data_sets([FLAGS.geno_file,FLAGS.pheno_file,FLAGS.trait_name], dummy_data = FLAGS.dummy_data)
    
   with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)   
       
    #placeholder for input data
    SNP_placeholder,trait_placeholder = placeholder_input(FLAGS.batch_size)
    
    #Build the graph to predict the trait
    trait_predict = deepgta_model.inference(SNP_placeholder)
    
    #Add loss op into the graph
    loss = deepgta_model.loss(trait_predict,trait_placeholder)
    
    #Add training op into graph
    train_step = deepgta_model.training(loss,global_step)
    
    #Add Op to compare the predict trait with real trait and compute R2
    eval_loss,total_error,unexplained_error = deepgta_model.evaluation(trait_predict,trait_placeholder)
    
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
    max_R_score = -float("inf");
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
            if R_squared_validation>=max_R_score:
                max_R_score = max(R_squared_validation,max_R_score)
                checkpoint_file = os.path.join(FLAGS.log_dir+'/model', 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=i)
    return R_squared_train,R_squared_validation,R_squared_test
        
def main(_):
    if not FLAGS.geno_file:
        raise ValueError("Must set --geno_file to the SNP file path, which need to be in the plink bed file formart.")
    if not FLAGS.pheno_file:
        raise ValueError("Must set --pheno_file to the trait file name.")
    if not FLAGS.trait_name:
        raise ValueError("Must set --trait_name to the expecting trait name.")
    R1,R2,R3 = run_training()
    R = [R1,R2,R3]
    with open(FLAGS.record_file,'w+') as f:
        f.write(','.join(str(r) for r in R)+'\n')
    
if __name__ == '__main__':
  tf.app.run()
    #FLAG_DICT = {input_data_dir,dummy_data,batch_size,learning_rate,l1_norm,max_step}
    