# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 20:45:07 2016

@author: lenovo
"""
import numpy as np
from scipy import sparse
from os import path
from matplotlib import pyplot as plt
def load():
    if not path.exists('data\ml-100k\u.data'):
        raise IOError("Data has not been downloaded.")
    data=np.loadtxt('data\ml-100k\u.data')
    ij=data[:,:2]
    ij-=1
    values=data[:,2]
    values=data[:,2]
    reviews=sparse.csc_matrix((values,ij.T)).astype(float)
    return reviews.toarray()

def all_correlations(bait,target):
    return np.array([np.corrcoef(bait,c)[0,1] for c in target])
    
def estimate(user,rest):
    bu=user>0
    br=rest>0
    ws=all_correlations(bu,br)
    selected=ws.argsort()[-100:]
    estimates=rest[selected].mean(0)
    estimates/=(.1+br[selected].mean(0))

def nn_movie(movie_likeness,reviews,uid,mid):
    likes=movie_likeness[mid].argsort()
    likes=likes[::-1]
    for ell in likes:
        if reviews[uid,ell]>0:
            return reviews[uid,ell]

def all_estimate(reviews,k=1):
    reviews=reviews.astype(float)
    k-=1
    nusers,nmovies=reviews.shape
    estimates=np.zeros_like(reviews)
    for u in range(nusers):
        ureviews=np.delete(reviews,u,axis=0)
        ureviews-=ureviews.mean(0)
        ureviews/=(ureviews.std(0)+1e-5)
        ureviews=ureviews.T.copy()
        
        for m in np.where(reviews[u]>0)[0]:
            estimates[u,m]=nn_movie(ureviews,reviews,u,m)
    return estimates



reviews = load()
#imagedata=reviews[:200,:200].todense()
#imagedata[imagedata>0]=1;
#
#plt.imshow(imagedata,interpolation='nearest',cmap ='gray')


