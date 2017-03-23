# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np, tensorflow as tf
import math,os
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
      'sim_record_dir',
      default_value=None,
      docstring='Record file name to store the simulation data',
  )
def GenPhenotype(n_sample = 5000, n_SNP = 1000, n_effectSNP = 100, hsq = 0.6, f = 0.5, effect_size = 0.1,indx = None):
    if type(f) is float:
        f = [f]
    if type(effect_size) is float:
        effect_size = [effect_size]
    assert len(f)==n_SNP or len(f)==1, "SNP frequency must have length equal to SNP number or 1."
    assert len(effect_size)==n_effectSNP or len(effect_size)==1,"SNP frequency must have length equal to observed SNP number or 1."
    if len(f)==1:
        f = f * n_SNP
    if len(effect_size)==1:
        effect_size = effect_size * n_effectSNP
    
    ### Genotype ###
    x = np.empty((n_sample, n_SNP))
    x[:] = np.NAN
    for i in range(n_SNP):
        x[:,i] = np.random.binomial(2, f[i],size = n_sample)
    
    # if observed SNP is not specified, choose them randomly
    if indx is None:
        indx = np.arange(n_SNP)
        np.random.shuffle(indx)
        indx = indx[0:n_effectSNP]
    
    ### Phenotype ###
    gy = np.dot(x[:,indx],effect_size)
    y = gy + np.random.normal( 0, math.sqrt(np.var(gy)*(1/hsq-1)),size = n_sample)
    y = (y - np.mean(y))/np.std(y)    
    return x, y,indx
def make_config(sample_size =5000, SNP_n = 1000,effectSNP_n = 100, hsq = 0.6,effect_std = 1):
    config = dummy_data_config
    config.individual_n = sample_size
    config.SNP_n = SNP_n
    config.effectSNP_n = effectSNP_n
    config.hsq = hsq
    config.frequency = np.random.uniform(0.01, 0.5, SNP_n)
    config.effect_size = np.random.normal(0,effect_std,effectSNP_n)
    indx = np.arange(SNP_n)
    np.random.shuffle(indx)
    indx = indx[0:effectSNP_n]
    config.index = indx
    return config
class dummy_data_config(object):
    """Dummy data configuration"""      
    individual_n = 5000
    SNP_n = 1000
    effectSNP_n = 100
    hsq = 0.6
    frequency = None
    effect_size = None
    index = None

def run_sim(config,keep_record_in = FLAGS.sim_record_dir):
    
    m = config.SNP_n
    mq = config.effectSNP_n
    n = config.individual_n
    f = config.frequency
    b = config.effect_size
    hsq = config.hsq
    indx = config.index
    #choose effect SNP index
    
    
    SNP,trait,_ = GenPhenotype(n,m,mq,hsq,f,b,indx)
    # make it a 2D tensor
    trait = [[x] for x in trait]
    trait = np.asarray(trait)
    if keep_record_in is not None:
     if not os.path.exists(keep_record_in):
         os.makedirs(keep_record_in)
     #Record SNP data
     with open(keep_record_in + '/train_SNP.dat','w+') as f:
      for i in range(n):
          f.write(','.join(str(x) for x in SNP[i,:])+'\n')
      f.close()
      
     #Record trait data
     with open(keep_record_in + '/train_trait.dat','w+') as f:
      f.write(','.join(str(x) for x in trait))
      f.close()
      
     #Record the index of effect SNP and the effect size
     with open(keep_record_in + '/effect_index.dat','w+') as f:
      f.write('#Index\n')
      f.write(','.join(str(x) for x in indx)+'\n')
      f.write('#Effect_size\n')
      f.write(','.join(str(x) for x in b)+'\n')
      f.close()
    return SNP, trait