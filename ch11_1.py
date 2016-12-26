# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 10:59:01 2016

@author: lenovo
"""
import numpy as np
from sklearn import linear_model,decomposition,datasets
from sklearn import lda 
from matplotlib import pylab
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D

#X=np.c_[np.ones(5),2*np.ones(5),10*np.ones(5)].T
#y=np.array([0,1,2])

iris = datasets.load_iris()
X = iris.data
y = iris.target

mds=manifold.MDS(n_components=3)
Xtrans=mds.fit_transform(X)


fig=pylab.figure(figsize=(10,4))
ax = fig.add_subplot(121, projection='3d')
ax.set_axis_bgcolor('white')

colors = ['r', 'g', 'b']
markers=['o',6,'*']
for c1,color,marker in zip(np.unique(y),colors,markers):
    ax.scatter(Xtrans[y==c1][:,0],Xtrans[y==c1][:,1],Xtrans[y==c1][:,2],c=color,marker=marker,edgecolor='black')
    
mds=manifold.MDS(n_components=2)
Xtrans=mds.fit_transform(X)
ax = fig.add_subplot(122)
for c1,color,marker in zip(np.unique(y),colors,markers):
    pylab.scatter(Xtrans[y==c1][:,0],Xtrans[y==c1][:,1],c=color,marker=marker,edgecolor='black')
    
