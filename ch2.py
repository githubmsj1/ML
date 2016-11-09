# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 10:46:25 2016

@author: lenovo
"""

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

data=load_iris()    
features=data['data']
target=data['target']
labels=np.array(['a'*10]*target.size)
for i in np.arange(3):
    labels[target==i]=data['target_names'][i]
#labels[target==0]=data['target_names'][0]

for t,marker,c in zip(xrange(3),">ox","rgb"):
    plt.scatter(features[target==t,0],
                features[target==t,1],
                marker=marker,
                c=c)
plt.xlabel("Sepel length(cm)")
plt.ylabel("Sepel width(cm)")
#plt.legend([data['target_names'][0],data['target_names'][1],data['target_names'][2]],loc="upper left")
plt.show()

for t,marker,c in zip(xrange(3),">ox","rgb"):
    plt.scatter(features[target==t,2],
                features[target==t,3],
                marker=marker,
                c=c)
plt.xlabel("Petal length(cm)")
plt.ylabel("Sepel width(cm)")
plt.show()

plength=features[:,2]
is_setosa=(labels=='setosa')

max_setosa=plength[is_setosa].max()
min_non_setosa= plength[~is_setosa].min()

print('Maximum of setosa: {0}.'.format(max_setosa))
print('Minimum of others: {0}.'.format(min_non_setosa))

features=features[~is_setosa]
labels=labels[~is_setosa]
virginica=(labels=='virginica')

best_acc=-1.0
for fi in xrange(features.shape[1]):
    thresh=features[:,fi].copy()
    thresh.sort()
    
    for t in thresh:
        pred=(features[:,fi]>t)
        acc=(pred==virginica).mean()
        if acc>best_acc:
            best_acc=acc
            best_fi=fi
            best_t=t

