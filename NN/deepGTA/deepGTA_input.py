#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:17:45 2017

@author: haotian.teng
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np, tensorflow as tf
import collections,sys
import plink_reader as reader
from plink_reader import extract_trait,extract_SNP
sys.path.append("/Users/haotian.teng/Documents/deepGTA/")
sys.path.append("/Users/haotian.teng/Documents/deepGTA/NN/Simulation")
import QTSim
from util.mutual_info import calc_MI
TRAIN_SNP = "/Users/haotian.teng/Documents/deepGTA/data/aric_hapmap3_m01_geno_ch22"
TRAIN_TRAIT ="/Users/haotian.teng/Documents/deepGTA/data/aric_outlier_117_u8682.txt"
TRAIT_NAME = "anta01"

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float(
      'jitter',
      default_value=None,
      docstring='If noe None, add noise to the training data.'
  )

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
extract_SNP_n = 400 #Only extract the first $effect_SNP_n SNPs with highest MI score. 

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
            assert SNP.shape[0]==trait.shape[0],"Individual number of SNP and trait should be same, SNP:%d , trait:%d"%(SNP.shape[0],trait.shape[0])
            self._SNP = SNP
            self._trait = trait
            self._individual_n = trait.shape[0]
            self._SNP_n = SNP.shape[1]
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
        if FLAGS.jitter is not None:
            std = np.std(self._trait)
            trait_size = len(trait_batch)
            noise = np.random.normal(0,std*FLAGS.jitter,trait_size)
            noise = [[x] for x in noise]
            trait_batch = trait_batch + noise
        return SNP_batch,trait_batch
            
def sig_index(SNP,trait,sig_n):
    bin_number = 5
    MI_list = np.empty(0)
    interval = (max(trait)-min(trait))/bin_number
    trait_bin = np.arange(min(trait),max(trait)+interval/2,interval)
    for index in range(SNP.shape[1]):
        MI_list = np.append(MI_list,calc_MI(SNP[:,index],trait,binsY = trait_bin))
    ind = sorted(range(len(MI_list)),key = lambda x: MI_list[x],reverse = True)[:sig_n]
    return ind
        
def read_data_sets(train_dir,
                   dummy_data = False
                   ):
    validation_size = NUM_EXAMPLES_PER_EPOCH_FOR_VALIDATION
    if dummy_data:
        config = QTSim.make_config(sample_size = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,SNP_n = 1000)
        def fake(config):
            return DataSet([],[],dummy_data = True,config = config)
        train = fake(config)
        #Set the sample size to validation_size
        config.individual_n = NUM_EXAMPLES_PER_EPOCH_FOR_VALIDATION
        validation = fake(config)
        config.individual_n = NUM_EXAMPLES_FOR_TEST
        test = fake(config)
        return Datasets(train = train,validation = validation,test = test)
    train_SNP = extract_SNP(TRAIN_SNP)
    train_trait = extract_trait(TRAIN_TRAIT,TRAIT_NAME)
    exclude_list = np.empty(0)
    for ind_index,current_trait in enumerate(train_trait):
        if np.isnan(current_trait[0]):
            exclude_list = np.append(exclude_list,ind_index)
    for ind_index,current_SNP in enumerate(train_SNP):
        if np.isnan(np.sum(current_SNP)):
            exclude_list = np.append(exclude_list,ind_index)
    train_SNP = train_SNP[~np.in1d(range(len(train_trait)),exclude_list)]
    train_trait = train_trait[~np.in1d(range(len(train_trait)),exclude_list)]
    train_trait = (train_trait - np.mean(train_trait))/np.std(train_trait)
    extract_index = sig_index(train_SNP,train_trait,extract_SNP_n)
    train_SNP = train_SNP[:,extract_index]
    #Seperate the dataset by validation size
    validation_SNP = train_SNP[:validation_size]
    validation_trait = train_trait[:validation_size]
    train_SNP = train_SNP[validation_size:]
    train_trait = train_trait[validation_size:]
    
    #Construct data set class
    train = DataSet(train_SNP,train_trait,dummy_data = False)
    validation = DataSet(validation_SNP,validation_trait,dummy_data = False)
    return Datasets(train = train,validation = validation,test = validation)