# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 10:19:23 2016

@author: lenovo
"""

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.linear_model import ElasticNet 
from sklearn.metrics import mean_squared_error,r2_score

boston=load_boston()
#plt.scatter(boston.data[:,5],boston.target,color='b')
#x=boston.data[:,5]
x=boston.data

#x=np.array([[v,1] for v in x])
#x=np.array([np.concatenate((v,[1])) for v in boston.data])
y=boston.target
#(slope,bias),total_error,_,_=np.linalg.lstsq(x,y)
s,total_error,_,_=np.linalg.lstsq(x,y)

#rmse=np.sqrt(total_error[0]/len(p))



#plt.plot(x,slope*x+bias,'-', color=(.9,.3,.3), lw=4)
#plt.plot(x,slope*x+bias,'-', color='r', lw=4)
#plt.xlim([3,12])
#plt.ylim([-10,60])
#plt.grid()


lr=LinearRegression(fit_intercept=True)#add a bias term
lr.fit(x,y)
#p=map(lr.predict,x)
p=lr.predict(x)
e=p-y
total_error=np.sum(e*e)
rmse_train=np.sqrt(total_error/len(p))
print('RMSE on training:{}'.format(rmse_train))


kf=KFold(len(x),n_folds=10)
err=0
for train,test in kf:
    lr.fit(x[train],y[train])
    p=lr.predict(x[test])
    e=p-y[test]
    err+=np.sum(e*e)
rmse_10cv=np.sqrt(err/len(x))
print('RMSE on 10-fold CV:{}'.format(rmse_10cv))


en=ElasticNet(fit_intercept=True,alpha=0.5)
kf=KFold(len(y),n_folds=5)
err=0
#pred=np.zeros_like(y)
for train,test in kf:
    en.fit(x[train],y[train])
    p=en.predict(x[test])

print('[EN 0.1] RMSE on testing (5 fold),{:.2}'.format(np.sqrt(mean_squared_error(y[test],p))))
print('[EN 0.1] RMSE on testing (5 fold),{:.2}'.format(r2_score(y[test],p)))
