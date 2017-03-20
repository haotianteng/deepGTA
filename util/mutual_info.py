#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 17:14:46 2017

@author: haotian.teng
"""

import numpy as np
def calc_MI(X,Y,binsX = None,binsY = None):
    c_X,_ = calc_hist(X,binsX)
    c_Y,_ = calc_hist(Y,binsY)
    c_XY = calc_hist2d(X,Y,binsX = binsX,binsY = binsY)
    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    return MI
def calc_hist(X,bins = None):
    if bins is None:
        bin_edges,c_X = np.unique(X,return_counts = True)
    else:
        c_X,bin_edges = np.histogram(X,bins)
    return c_X,bin_edges

def calc_hist2d(X,Y,binsX = None,binsY = None):
    if (binsX is None) and (binsY is None): 
        #X is a categorical variable,Y is a continous variable
        X_cat = np.unique(X)
        Y_cat = np.unique(Y)
        c_XY = np.zeros((len(X_cat),len(Y_cat)))
        for X_index,category in enumerate(X_cat):
            Y_hat = Y[np.in1d(X,category)]
            current_prop,current_category = calc_hist(Y_hat)
            index_XY = np.in1d(Y_cat,current_category)
            c_XY[X_index][index_XY] = current_prop
    if (binsX is None) and (binsY is not None):
        X_cat = np.unique(X)
        c_XY  = np.zeros((len(X_cat),len(binsY)-1))
        for X_index,category in enumerate(X_cat):
            Y_hat = Y[np.in1d(X,category)]
            current_prop,_ = calc_hist(Y_hat,bins = binsY)
            c_XY[X_index] = current_prop
    if (binsY is None) and (binsX is not None):
        Y_cat = np.unique(Y)
        c_XY = np.zeros(len(binsX)-1,len(Y_cat))
        for Y_index,category in enumerate(Y_cat):
            X_hat = X[np.in1d(Y,category)]
            current_prop,_ = calc_hist(X_hat,bins = binsX)
            c_XY[:,Y_index] = current_prop
    if (binsX is not None) and (binsY is not None):
        c_XY = np.histogram2d(X,Y,bins = (binsX,binsY))[0]
    return c_XY 

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H

#"""test code"""
#for bin_number in range(2,10):
#    config = QTSim.make_config()
#    SNP,trait = QTSim.run_sim(config,keep_record_in = None)
#    MI_list = np.empty(0)
#    interval = (max(trait)-min(trait))/bin_number
#    trait_bin = np.arange(min(trait),max(trait)+interval/2,interval)
#    for index in range(SNP.shape[1]):
#        MI_list = np.append(MI_list,calc_MI(SNP[:,index],trait,binsY = trait_bin))
#        SKLMI_list = np.append(SKLMI_list,mutual_info_score(SNP[:,index],trait[:,0]))
#    MI_list_effect = exclude(MI_list, config.index)
#    effect_correlation = np.corrcoef(MI_list[config.index],abs(config.effect_size))[0,1]
#    false_pos = np.sum(np.abs(MI_list_effect))
#    print("Correlation %4.2f"%(effect_correlation))
#    print("False positive: %4.2f"%(false_pos))
#    print("Bin number: %d"%bin_number)
#    print("++++++++++++++++++++++++++++")