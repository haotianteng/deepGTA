#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 16:44:06 2017

@author: haotian.teng
"""
import numpy as np

def exclude(a,exclude_list):
    b = [x for i,x in enumerate(a) if i not in exclude_list]
    b = np.asarray(b)
    return b