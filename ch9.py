# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 14:32:39 2016

@author: lenovo
"""

import scipy
from matplotlib.pyplot import specgram
sample_rate,X=scipy.io.wavfile.read("D:\Sundries\ML\data\genres\blues.00000.au")
print sample_rate,X.shape
specgram(X,Fs=sample_rate,xextent=(0,30))