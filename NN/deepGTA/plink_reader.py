#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:17:30 2017

@author: haotian.teng
"""
import math,os
import numpy as np

def extract_SNP(file_path,chrom_list = None):
    """Extract the SNP into a 2D numpy array [individual_n, SNP_n]
    Args:
        file_path: SNP file path,required *.bim,*.fam and *.bed file
        chrom_list: specific chromesome list, only extract SNP in these chromesome
    Returns:
        SNP_data: numpy SNP array
        
    """
    bim_file_path = file_path + ".bim"
    fam_file_path = file_path + ".fam"
    bed_file_path = file_path + ".bed"
    SNP_list = list()
    chrom_record = list()
    SNP_n= 0
    sample_size = 0
    with open(bim_file_path,'r') as bim_f:
        for line in bim_f:
            SNP_n +=1
            SNP_line_info = line.split()
            chrom_record.append(int(SNP_line_info[0]))
    with open(fam_file_path,'r') as fam_f:
        for line in fam_f:
            sample_size +=1
    print("sample size = %d"%(sample_size))
    print("SNP number = %d"%(SNP_n))
    SNP_index_list = None
    if chrom_list is not None:
        SNP_index_list = [i for i, x in enumerate(chrom_record) if x in chrom_list]
    block_size = math.ceil(sample_size / 4)
    bed_f = os.open(bed_file_path,os.O_RDONLY)
    magic_num = os.read(bed_f,3)
    magic_num = [format(x,'b') for x in magic_num]
    if not (magic_num[0] == "1101100") & (magic_num[1] == "11011") & (magic_num[2] == "1"):
        print ("File format does not support, magic number is not correct.")
    for SNP_index in range(min(1000,SNP_n)):
        if (SNP_index_list is None) or (SNP_index in SNP_index_list):
            SNP = np.empty(0)
            block = os.read(bed_f,block_size)
            byte_list = [format(x ,'b') for x in block]
            for byte_string in byte_list:
                SNP  = np.concatenate( (SNP,byte2SNP(byte_string)) )
            SNP_list.append(SNP)
    SNP_data = np.stack(SNP_list,axis = -1)
    return SNP_data[:sample_size]

def extract_trait(file_path,trait_name):
    """Extract the trait into a 1D numpy array [individual_n]
    Args:
        file_path: Information file for the individual.
        trait_name: trait name which need to be extracted, must be a single string.
    Returns:
        trait_data: numpy triat array to address the phenotype
        
    """
    trait_list = list()
    g = lambda x: [float(x)] if x!='NA' else [float('nan')]
    with open(file_path,'r') as trait_f:
        annotation = trait_f.readline().split()
        indx = annotation.index(trait_name)
        for line in trait_f:
            trait_list.append([g(x) for x in line.split() ][indx])
    trait_data = np.stack(trait_list,axis = 0)
    return trait_data
    
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

        
    
#SNP_record = extract_SNP("/Users/haotian.teng/Documents/deepGTA/data/aric_hapmap3_m01_geno",chrom_list = [1])
trait_record =extract_trait("/Users/haotian.teng/Documents/deepGTA/data/aric_outlier_117_u8682.txt",trait_name = 'anta01')