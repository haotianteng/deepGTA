# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np, statistics as sta, matplotlib.pyplot as plt
import math,os

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
    y = gy + np.random.normal( 0, math.sqrt(sta.variance(gy)*(1/hsq-1)),size = n_sample)
    y = (y - np.mean(y))/np.std(y)    
    return x, y,indx

### Parameter setting
n1 = 5000 #Sample number
n2 = 3000
m = 1000 #SNP number
mq = 100 #Number of observable SNP
f = np.random.uniform( 0.01, 0.5, m) #SNP proportion
b = np.random.normal(0, 1,mq) #Effect size
hsq = 0.6

indx = np.arange(m)
np.random.shuffle(indx)
indx = indx[0:mq]
data_folder = os.getcwd()+'/data/'
# training sets
x1,y1,_ = GenPhenotype(n1,m,mq,hsq,f,b,indx)
train_gene = open(data_folder + 'train_geno.dat','w+')
train_pheno = open(data_folder + 'train_pheno.dat','w+')
for i in range(n1):
    train_gene.write(','.join(str(x) for x in x1[i,:])+'\n')
train_pheno.write(','.join(str(x) for x in y1))
train_gene.close()
train_pheno.close()
# testing set
x2,y2,_ = GenPhenotype(n2,m,mq,hsq,f,b,indx)
test_gene = open(data_folder + 'test_gene.dat','w+')
test_pheno = open(data_folder + 'test_pheno.dat','w+')
for i in range(n2):
    test_gene.write(','.join(str(x) for x in x2[i,:])+'\n')
test_pheno.write(','.join(str(x) for x in y2))
test_gene.close()
test_pheno.close()
corr_test = np.empty(mq)
for i in range(mq):
    corr = np.corrcoef(x1[:,indx[i]],y1)
    corr_test[i] = corr[0,1]
plt.plot(b,corr_test,'ro')
plt.show()

effect_index = open(data_folder + 'effect_index.dat','w+')
effect_index.write('#Index\n')
effect_index.write(','.join(str(x) for x in indx)+'\n')
effect_index.write('#Effect_size\n')
effect_index.write(','.join(str(x) for x in b) + '\n')
effect_index.close()
