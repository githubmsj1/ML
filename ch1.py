# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 23:24:01 2016

@author: lenovo
"""
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def error(f,x,y):
    return sum((f(x)-y)**2)
    
    


data=sp.genfromtxt("D:\Sundries\ML\web_traffic.tsv",delimiter="\t")
x=data[:,0]
y=data[:,1]
x=x[~sp.isnan(y)]
y=y[~sp.isnan(y)]

plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w*7*24 for w in range(10)],['week %i'%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()



"""
    curve fitting for 1 and 2 order
"""
fp1,res,rank,sv,rcond=sp.polyfit(x,y,1,full=True)
print ("Model parameter: %s"%fp1)
print (res)
f1=sp.poly1d(fp1)
print (error(f1,x,y))

fx=sp.linspace(0,x[-1],1000)
plt.plot(fx,f1(fx),linewidth=4);
plt.legend(["d=%i"%f1.order],loc="upper left")

fp2=sp.polyfit(x,y,2)
print (fp2)
f2=sp.poly1d(fp2)
print(error(f2,x,y))
plt.plot(fx,f2(fx),linewidth=4);
#plt.legend(["d1=%i"%f1.order,"d2=%i"%f2.order],loc="upper left")


inflection=3.5*7*24
xa=x[:inflection]
ya=y[:inflection]
xb=x[inflection:]
yb=y[inflection:]

fa=sp.poly1d(sp.polyfit(xa,ya,1))
fb=sp.poly1d(sp.polyfit(xb,yb,1))
faError=error(fa,xa,ya)
fbError=error(fb,xb,yb)
print("Error inflection=%f"%(faError+fbError))
plt.plot(xa,fa(xa),linewidth=4);
plt.plot(xb,fb(xb),linewidth=4);
plt.legend(["d=%d"%f1.order,"d=%d"%f2.order,"d=%d"%fa.order,"d=%d"%fb.order],loc="upper left")
plt.show()

reached_max=fsolve(f2-100000,800)/(7*24)
print("100000 hits/hour expected at week %f"%reached_max)




    