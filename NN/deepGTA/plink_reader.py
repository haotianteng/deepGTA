#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:17:30 2017

@author: haotian.teng
"""
import math,os
import numpy as np

def read_geno(file_path,chrom_list = None):
    bim_file_path = file_path + ".bim"
    fam_file_path = file_path + ".fam"
    bed_file_path = file_path + ".bed"
    SNP_list = list()
    SNP_n= 0
    sample_size = 0
    with open(bim_file_path,'r') as bim_f:
        for line in bim_f:
            SNP_n +=1
    with open(fam_file_path,'r') as fam_f:
        for line in fam_f:
            sample_size +=1
    print("sample size = %d"%(sample_size))
    print("SNP number = %d"%(SNP_n))
    block_size = math.ceil(sample_size / 4)
    bed_f = os.open(bed_file_path,os.O_RDONLY)
    magic_num = os.read(bed_f,3)
    magic_num = [format(x,'b') for x in magic_num]
    if not (magic_num[0] == "1101100") & (magic_num[1] == "11011") & (magic_num[2] == "1"):
        print ("File format does not support, magic number is not correct.")
    for SNP_index in range(min(1000,SNP_n)):
        block = os.read(bed_f,block_size)
        SNP_list.append([format(x ,'b') for x in block])
    return SNP_list

def byte2SNP(byte_string):
    SNP = np.zeros(4)
    length = len(byte_string)
    SNP_index = 0
    SNP[SNP_index] = bit2SNP(byte_string[-2:])
    SNP_index+=1
    for i in range(-2,-length,-2):
        SNP[SNP_index] = bit2SNP(byte_string[i-2:i])
        SNP_index+=1
    return SNP    
def bit2SNP(bit):
    """Transfer 2-bit into the correspond meaning, check the plink website for 
    detail information, transfer dictonary:
        00	0        Homozygous for first allele in .bim file
        01   np.nan   Missing genotype
        10	1        Heterozygous
        11	2        Homozygous for second allele in .bim file
    """
    if bit=='00':
        return 0
    elif bit =='01':
        return np.nan
    elif bit =='10':
        return 1
    elif bit =='11':
        return 2
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
    
SNP_record = read_geno("/Users/haotian.teng/Documents/deepGTA/NN/data/aric_hapmap3_m01_geno")