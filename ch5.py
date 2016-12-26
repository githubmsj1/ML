# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 14:54:16 2016

@author: lenovo
"""

import os
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report

def lr_model(clf,x):
    return 1/(1+np.exp(-(clf.intercept_+clf.coef_*x)))

np.random.seed(3)
num_per_class=40
X=np.hstack((norm.rvs(2,size=num_per_class,scale=2),
             norm.rvs(8,size=num_per_class,scale=3)))
y=np.hstack((np.zeros(num_per_class),
             np.ones(num_per_class)))

clf=LogisticRegression()
clf.fit(X.reshape(num_per_class * 2, 1),y)
print(np.exp(clf.intercept_),np.exp(clf.coef_.ravel()))
print("P(x=-1)=%.2f\tP(x=7)=%.2f"%(lr_model(clf,-1),lr_model(clf,7)))

X_test=np.arange(-5,20,0.1)
precision, recall, thresholds = precision_recall_curve(y,clf.predict(X.reshape(num_per_class * 2, 1)))

probs_for_good=clf.predict_proba(X.reshape(num_per_class*2,1))[:,1]
print(classification_report(clf.predict(X.reshape(num_per_class * 2, 1)),clf.predict_proba(X.reshape(num_per_class * 2, 1))[:,1]>0.63,
                            target_names=['not accpeted','accepted']))