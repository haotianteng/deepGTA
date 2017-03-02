#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:17:45 2017

@author: haotian.teng
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import collections
import QTSim
DATA_DIR = ""
TRAIN_SNP = "train_geno.dat"
TRAIN_TRAIT = "train_pheno.dat"

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
    
def extract_SNP(f):
    """Extract the SNP into a 2D numpy array [individual_n, SNP_n]
    Args:
        f: SNP file handle
    Returns:
        SNP_data: numpy SNP array
        
    """
    SNP_data = list()
    for line in f:
        SNP_data.append([float(x) for x in line.split(',')])
    SNP_data = np.asarray(SNP_data)
    return SNP_data

def extract_trait(f):
    """Extract the trait into a 1D numpy array [individual_n]
    Args:
        f: SNP file handle
    Returns:
        trait_data: numpy triat array to address the phenotype
        
    """
    trait_data = list()
    for line in f:
        trait_data.append([float(x) for x in line.split(',')])
    trait_data = np.asarray([[x] for x in trait_data[0]])
    return trait_data

class DataSet(object):
    def __init__(self,
                 SNP,
                 trait,
                 dummy_data = False,
                 ):
        """Custruct a DataSet."""
        if dummy_data:
            config = QTSim.dummy_data_config
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
          self._SNP = self.images[perm0]
          self._trait = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._individual_n:
          # Finished epoch
          self._epochs_completed += 1
          # Get the rest samples in this epoch
          rest_individual_n = self._individual_n - start
          images_rest_part = self._SNP[start:self._individual_n]
          labels_rest_part = self._trait[start:self._individual_n]
          # Shuffle the data
          if shuffle:
            perm = np.arange(self._individual_n)
            np.random.shuffle(perm)
            self._SNP = self.images[perm]
            self._trait = self.labels[perm]
          # Start next epoch
          start = 0
          self._index_in_epoch = batch_size - rest_individual_n
          end = self._index_in_epoch
          images_new_part = self._SNP[start:end]
          labels_new_part = self._trait[start:end]
          return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
          self._index_in_epoch += batch_size
          end = self._index_in_epoch
          return self._SNP[start:end], self._trait[start:end]
            
        
def read_data_sets(train_dir,
                   validation_size = 1000,
                   dummy_data = False
                   ):
    if dummy_data:
        def fake():
            return DataSet([],[],dummy_data = True)
        
        train = fake()
        
        #Set the sample size to validation_size
        QTSim.dummy_data.config.individual_n = validation_size
        validation = fake()
        
        return Datasets(train = train,validation = validation)
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