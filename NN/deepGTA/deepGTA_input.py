#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:17:45 2017

@author: haotian.teng
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np, tensorflow as tf,plink_reader as reader
import collections
import sys
sys.path.append("/Users/haotian.teng/Documents/deepGTA/NN/Simulation")
import QTSim
DATA_DIR = ""
TRAIN_SNP = "train_geno.dat"
TRAIN_TRAIT = "train_pheno.dat"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean(
      'jitter',
      default_value=False,
      docstring='If true, add noise to the training data.'
  )

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
SNP_n = 1000

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5000
NUM_EXAMPLES_PER_EPOCH_FOR_VALIDATION = 1000
NUM_EXAMPLES_FOR_TEST = 1000


class DataSet(object):
    def __init__(self,
                 SNP,
                 trait,
                 dummy_data = False,
                 config = None #dummy data configuration
                 ):
        """Custruct a DataSet."""
        if dummy_data:
            if config is None:
                raise NameError('Configuration for dummy_data missing!')
            self._individual_n = config.individual_n
            self._SNP_n = config.SNP_n
            self._SNP,self._trait = QTSim.run_sim(config)
        else:
            assert SNP.shape[0]==trait[0],"Individual number of SNP and trait should be same, SNP:%d , trait:%d"%(SNP.shape[0],trait[0])
            self._SNP = SNP
            self._trait = trait
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
    @property
    def SNP(self):
        return self._SNP
    
    @property
    def trait(self):
        return self._trait
    
    @property
    def individual_n(self):
        return self._individual_n
    
    @property
    def SNP_n(self):
        return self._SNP_n
    
    @property 
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size,shuffle = True):
        """Return next batch in batch_size from the data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
          perm0 = np.arange(self._individual_n)
          np.random.shuffle(perm0)
          self._SNP = self.SNP[perm0]
          self._trait = self.trait[perm0]
        # Go to the next epoch
        if start + batch_size > self._individual_n:
          # Finished epoch
          self._epochs_completed += 1
          # Get the rest samples in this epoch
          rest_individual_n = self._individual_n - start
          SNP_rest_part = self._SNP[start:self._individual_n]
          trait_rest_part = self._trait[start:self._individual_n]
          # Shuffle the data
          if shuffle:
            perm = np.arange(self._individual_n)
            np.random.shuffle(perm)
            self._SNP = self.SNP[perm]
            self._trait = self.trait[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size - rest_individual_n
          end = self._index_in_epoch
          SNP_new_part = self._SNP[start:end]
          trait_new_part = self._trait[start:end]
          SNP_batch = np.concatenate((SNP_rest_part, SNP_new_part), axis=0)
          trait_batch = np.concatenate((trait_rest_part, trait_new_part), axis=0)
        else:
          self._index_in_epoch += batch_size
          end = self._index_in_epoch
          SNP_batch = self._SNP[start:end]
          trait_batch = self._trait[start:end]
        if FLAGS.jitter:
            std = np.std(self._trait)
            trait_size = len(trait_batch)
            noise = np.random.normal(0,std/2,trait_size)
            noise = [[x] for x in noise]
            trait_batch = trait_batch + noise
        return SNP_batch,trait_batch
            
        
def read_data_sets(train_dir,
                   dummy_data = False
                   ):
    validation_size = NUM_EXAMPLES_PER_EPOCH_FOR_VALIDATION
    if dummy_data:
        config = QTSim.make_config(sample_size = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,SNP_n = SNP_n)
        def fake(config):
            return DataSet([],[],dummy_data = True,config = config)
        train = fake(config)
        #Set the sample size to validation_size
        config.individual_n = NUM_EXAMPLES_PER_EPOCH_FOR_VALIDATION
        validation = fake(config)
        config.individual_n = NUM_EXAMPLES_FOR_TEST
        test = fake(config)
        return Datasets(train = train,validation = validation,test = test)
    with open(TRAIN_SNP,'r') as f:
        train_SNP = extract_SNP(f)
    with open(TRAIN_TRAIT,'r') as f:
        train_trait = extract_trait(f)
        
    #Seperate the dataset by validation size
    validation_SNP = train_SNP[:validation_size]
    validation_trait = train_trait[:validation_size]
    train_SNP = train_SNP[validation_size:]
    train_trait = train_trait[validation_size:]
    
    #Construct data set class
    train = DataSet(train_SNP,train_trait,dummy_data = dummy_data)
    validation = DataSet(validation_SNP,validation_trait,dummy_data = dummy_data)
    return Datasets(train = train,validation = validation)