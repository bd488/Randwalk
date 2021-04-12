# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 19:35:53 2021

@author: 44751
"""

from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from matplotlib import pyplot as plt
from matplotlib import rcParams

import pylab as plb

colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")

# Define parameters for the walk#
x=0
fig = plt.figure(figsize=(8,4),dpi=200)
ax = fig.add_subplot(111)
dims = 1
step_n = 10000
step_set = [-1, 0, 1]


arr = []

while x<100000:#<---------- number of runs    
    origin = np.zeros((1,dims))
    # Simulate steps in 1D
    step_shape = (step_n,dims)
    steps = np.random.choice(a=step_set, size=step_shape, p=[1/3,1/3,1/3] ) #<----------- probabilties of moving in given direction, uneven to simulate field, feel free to play with them.
    path = np.concatenate([origin, steps]).cumsum(0)
    start = path[:1]
    stop = path[-1:]
    # Plot the path
    p2 = int(path[step_n - 1])
    arr.append(p2)
    #print(arr)
    
    #ax.scatter(np.arange(step_n+1), path, c="black",alpha=0.25,s=0.05);
    #ax.plot(path,c="black",alpha=0.5,lw=0.5,ls="-",);
    #ax.plot(0, start, c="red", marker="+")
    #ax.plot(step_n, stop, c="black", marker="o")
    #ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.tick_params(axis='both', which='minor', labelsize=14)
   
    plt.tight_layout(pad=0)
    x=x+1
    ax.set_xlabel('Time', fontsize = 16)
    ax.set_ylabel('Displacement', fontsize = 16)

    
fig2 = plt.figure(figsize=(8,8),dpi=200)
ax2 = fig2.add_subplot(111)
#print(arr)
arr = np.asarray(arr)
freq=[]
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='minor', labelsize=14)



xn = arr
for value in np.nditer(arr): 
    occurrences = np.count_nonzero(arr == value)
    freq.append(occurrences)
    #print(occurrences)
    #print(value)
    ax2.plot(value, occurrences, marker = 'x',color = 'black')
ax2.set_xlabel('Displacement ', fontsize = 16)
ax2.set_ylabel('Frequency', fontsize = 16)


mean = np.mean(arr)
#print(mean)
sigma = np.std(arr)

#print(len(value2))
#print(len(arr))

oc_set = set()
res = []
for idx, val in enumerate(freq):
    if val not in oc_set:
        oc_set.add(val)         
    else:
        res.append(idx)
        
#print(res)

np.delete(arr,res)

                     
def gaus(xn,a,xn0,sigma):                      
    return a*exp(-(xn-xn0)**2/(2*sigma**2))

popt,pcov = curve_fit(gaus,arr,freq,p0=[1,mean,sigma])
ax2.plot(xn,gaus(xn,*popt),'ro:',label='fit', linestyle='', color = 'black')
#plt.savefig("plots/random_walk_1d.png",dpi=250);

