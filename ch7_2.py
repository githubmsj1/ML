# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 19:39:03 2016

@author: lenovo
"""

from sklearn.linear_model import LassoCV

reg=LassoCV(fit_intercept=True,alphas=[0.125,0.25,0.5,1.,2.,4.])
