# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 10:10:41 2016

@author: lenovo
"""

import numpy as np
from collections import defaultdict
from itertools import chain
from gzip import GzipFile


def rules_from_itemset(itemset,dataset,minlift=1.):

    nr_transactions=float(len(dataset))
    for item in itemset:
        consequent=frozenset([item])
        antecedent=itemset-consequent
        base=0.0
        
        acount=0.0
        
        ccount=0.0
        for d in dataset:
            if item in d:base+=1
            if d.issuperset(itemset):ccount+=1
            if d.issuperset(antecedent):acount+=1
        base/=nr_transactions
        p_y_given_x=ccount/acount
        lift=p_y_given_x/base
        if lift>minlift:
            print('Rule{0}->{1}has lift {2}'.format(antecedent,consequent,lift))

dataset =[[int(tok) for tok in line.strip().split()] for line in open('retail.dat')]
counts=defaultdict(int)
for elem in chain(*dataset):
    counts[elem]+=1

minsupport=80

valid=set(el for el,c in counts.items() if (c>=minsupport))

dataset=[[el for el in ds if(el in valid)] for ds in dataset]
dataset=[frozenset(ds) for ds in dataset]#for fast

itemsets=[frozenset([v]) for v in valid]
freqsets=itemsets[:]

tested=set()
nextsets=[]

for it in itemsets:
    for v in valid:
        if v not in it:
            c=(it |frozenset([v]))
            
            if c in tested:
                continue
            tested.add(c)
            
            support_c=sum(1 for d in dataset if d.issuperset(c))
            if support_c>minsupport:
                nextsets.append(c)
freqsets.extend(nextsets)
itemsets=nextsets