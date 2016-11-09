# -*- coding: utf-8 -*-
"""
Created on Sun Nov 06 10:24:51 2016

@author: lenovo
"""

import os 
import scipy as sp
import sys
from scipy.stats import norm
from matplotlib import pylab
from sklearn.cluster import KMeans

DATA_DIR=os.path.join(os.path.dirname(os.path.realpath(__file__)),'data')
CHART_DIR=os.path.join(os.path.dirname(os.path.realpath(__file__)),'charts')

if not os.path.exists(DATA_DIR):
    sys.abort

if not os.path.exists(CHART_DIR):
    os.mkdir(CHART_DIR)

seed=2
sp.random.seed(seed)

num_clusters=3

def plot_clustering(x,y,title,mx=None,ymax=None,km=None):
    pylab.figure(num=None,figsize=(8,6))
    if km:
        pylab.scatter(x,y,s=50,c=km.predict(list(zip(x,y))))
    else:
        pylab.scatter(x,y,s=50)
    pylab.title(title)
    pylab.xlabel("word 1")
    pylab.ylabel("word 2")
    
    pylab.autoscale(tight=True)
    pylab.ylim(ymin=0,ymax=1)
    pylab.xlim(xmin=0,xmax=1)
    pylab.grid(True,linestyle='-',color='0.75')
    
    return pylab


xw1=norm(loc=0.7,scale=0.15).rvs(20)
yw1=norm(loc=0.3,scale=0.15).rvs(20)


xw2=norm(loc=0.7,scale=0.15).rvs(20)
yw2=norm(loc=0.7,scale=0.15).rvs(20)

xw3=norm(loc=0.2,scale=0.15).rvs(20)
yw3=norm(loc=0.8,scale=0.15).rvs(20)

x=sp.append(sp.append(xw1,xw2),xw3)
y=sp.append(sp.append(yw1,yw2),yw3)

i=1
plot_clustering(x,y,"Vectors")
pylab.savefig(os.path.join(CHART_DIR,"%i.png")%i)
pylab.clf()
i+=1

mx, my=sp.meshgrid(sp.arange(0,1,0.001),sp.arange(0,1,0.001))

km=KMeans(init='random',n_clusters=num_clusters,verbose=1,
          n_init=1,max_iter=1,random_state=seed)
km.fit(sp.array(list(zip(x,y))))

Z=km.predict(sp.c_[mx.ravel(),my.ravel()]).reshape(mx.shape)

plot_clustering(x,y,"Clustering iteration 1",km=km)
pylab.imshow(Z,interpolotion='nearest',
             extent=(mx.min(),mx.max(),my.min(),my.max()),
             cmpa=pylab.cm.Blues,
             aspect='auto',origin='lowers')

c1a,c1b,c1c=km.cluster_centers_