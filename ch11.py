# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 10:59:01 2016

@author: lenovo
"""
import numpy as np
from sklearn import linear_model,decomposition,datasets
from sklearn import lda 
from matplotlib import pylab
x1=np.arange(0,10,0.2)
x2=x1+np.random.normal(loc=0,scale=1,size=len(x1))
X=np.c_[(x1,x2)]
        
##case 1 PCA
#good=(x1>5)|(x2>5)
#bad=~good
#
#pca=decomposition.PCA(n_components=1)
#Xtrans=pca.fit_transform(X)
#print(pca.explained_variance_ratio_)
#pylab.clf()
#fig=pylab.figure(num=None,figsize=(10,4))
#pylab.subplot(121)
#pylab.xlabel("$X_1$")
#pylab.ylabel("$X_2$")
#pylab.scatter(x1[good],x2[good],edgecolors="blue",facecolor="blue")
#pylab.scatter(x1[bad],x2[bad],edgecolors="red",facecolor="white")
#pylab.grid(True)
#
#pylab.subplot(122)
#pylab.xlabel("$X'$")
#pylab.scatter(Xtrans[good],np.zeros(len(Xtrans[good])),edgecolors="blue",facecolor="blue")
#pylab.scatter(Xtrans[bad],np.zeros(len(Xtrans[bad])),edgecolors="red",facecolor="white")
#pylab.grid(True)
#pylab.autoscale(tight=True)


##case 2 LDA
good=x1>x2
bad=~good
lda_inst=lda.LDA(n_components=1)
Xtrans=lda_inst.fit_transform(X,good)


pylab.clf()
fig=pylab.figure(num=None,figsize=(10,4))
pylab.subplot(121)
pylab.xlabel("$X_1$")
pylab.ylabel("$X_2$")
pylab.scatter(x1[good],x2[good],edgecolors="blue",facecolor="blue")
pylab.scatter(x1[bad],x2[bad],edgecolors="red",facecolor="white")
pylab.grid(True)

pylab.subplot(122)
pylab.xlabel("$X'$")
pylab.scatter(Xtrans[good],np.zeros(len(Xtrans[good])),edgecolors="blue",facecolor="blue")
pylab.scatter(Xtrans[bad],np.zeros(len(Xtrans[bad])),edgecolors="red",facecolor="white")
pylab.grid(True)
pylab.autoscale(tight=True)
