# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 08:41:04 2016

@author: lenovo
"""

from gensim import corpora,models,similarities
import numpy as np
from scipy.spatial import distance

def closest_to(doc_id):
    return pairwise[doc_id].argmin()


#corpus=corpora.BleiCorpus('./ap/ap.dat','./ap/vocab.txt')
#
#model=models.ldamodel.LdaModel(
#corpus,
#num_topics=100,
#id2word=corpus.id2word)
#
#topics=[model[c] for c in corpus]
#hdp=models.hdpmodel.HdpModel(corpus,corpus.id2word)##Don't need to specify number of topic

dense=np.zeros((len(topics),100),float)
for ti,t in enumerate(topics):
    for tj,v in t:
        dense[ti,tj]=v

pairwise=distance.squareform(distance.pdist(dense))

largest=pairwise.max()
for ti in range(len(topics)):
    pairwise[ti,ti]=largest+1



