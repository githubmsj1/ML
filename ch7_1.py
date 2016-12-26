# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 19:07:20 2016

@author: lenovo
"""

from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.cross_validation import KFold
import numpy as np

data,target=load_svmlight_file('data/E2006.train')
print('Min target value"{}'.format(target.min()))

lr=LinearRegression(fit_intercept=True)
lr.fit(data,target)

p=np.array(lr.predict(data))
p=p.ravel()

e=p-target
total_sq_error=np.sum(e*e)
rmse_train=np.sqrt(total_sq_error/len(p))
print(rmse_train)

met=ElasticNetCV(fit_intercept=True)
kf=KFold(len(target),n_folds=10)
for train,test in kf:
    met.fit(data[train],target[train])
    p=met.predict(data[test])
    p=np.array(p).ravel()
    e=p-target[test]
    err+=np.dot(e,e)
rmse_10cv=np.sqrt(err/len(target))