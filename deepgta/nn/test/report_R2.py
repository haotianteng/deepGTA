#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:07:24 2017

@author: haotian.teng
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,argparse,sys
import numpy as np,tensorflow as tf,matplotlib.pyplot as plt

def collect_variance_result(record_dir,base_name,plot = False):
    """Collect the R_squared after runing the deepGTA in dummy data"""
    r = list()
    for file in sorted(os.listdir(record_dir)):
        if file.startswith(base_name):
            file_name = os.path.join(record_dir,file)
            with open(file_name) as f:
                for line in f:
                    r.append([float(x) for x in line.split(',')])
    r = np.asarray(r)
    return r

def main(_):
    r_record = collect_variance_result(FLAGS.record_dir,FLAGS.base_name)
    r_train = r_record[:,0]
    r_valid = r_record[:,1]
    r_test = r_record[:,2]
    print(np.mean(r_valid))
    print(np.mean(r_test))
    plt.figure()
    plt.hist(r_train)
    plt.hist(r_valid)
   # plt.hist(r_test)
    plt.show()
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
          '--record_dir',
          type = str,
          default = "/Users/haotian.teng/Documents/deepGTA/NN/Test/record",
          help = "Record directory.")
  parser.add_argument(
          '--base_name',
          type = str,
          default = "R_squared_record",
          help = "Prefix of the file name.")
  FLAGS,unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)