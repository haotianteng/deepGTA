#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 17:14:46 2017

@author: haotian.teng
"""

import numpy as np
from plink_reader import extract_SNP,extract_trait

def calc_MI(X,Y,binsX = None,binsY = None):
    c_X = calc_prop(X,binsX)
    c_Y = calc_prop(Y,binsY)
    
    c_XY = np.histogram2d(X,Y,bins)[0]

       H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    return MI
def calc_prop(X,bins = None):
    if bins = None:
        c_X = np.unique(X,return_counts = True)
    else:
        c_X = np.histogram(X,bins)[0]
    return c_X
def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H
