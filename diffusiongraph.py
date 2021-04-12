# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 22:21:05 2021

@author: 44751
"""

from itertools import cycle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import pandas as pd
import pylab as plb



data = pd.read_csv("data2.csv")
fig = plt.figure(figsize=(8,4),dpi=200)
ax=fig.add_subplot(111)






##############################################################1/V##################################################


T1 = data.iloc[:,0]
D1 = data.iloc[:,1]


T1 = np.array(T1)
D1 = np.array(D1)


T1 = T1[~pd.isnull(T1)]
D1 = D1[~pd.isnull(D1)]



T1 = T1**(1.5)

T2 = data.iloc[:,2]
D2 = data.iloc[:,3]


T2 = np.array(T2)
D2 = np.array(D2)


T2 = T2[~pd.isnull(D2)]
D2 = D2[~pd.isnull(D2)]



T2 = T2**(1.5)

T3 = data.iloc[:,4]
D3 = data.iloc[:,5]


T3 = np.array(T3)
D3 = np.array(D3)


T3 = T3[~pd.isnull(D3)]
D3 = D3[~pd.isnull(D3)]





T3 = T3**(1.5)

T4 = data.iloc[:,6]
D4 = data.iloc[:,7]


T4 = np.array(T4)
D4 = np.array(D4)


T4 = T4[~pd.isnull(D4)]
D4 = D4[~pd.isnull(D4)]



T4 = T4**(1.5)
print(T1)



#################################################################PLOTTING############################################


#ax.plot(T1, D1, color = 'black', marker = 'o', linestyle ='')


def give_me_a_straight_line(T1,D1):
    w, b  = np.polyfit(T1,D1,deg=1)
    line  = w * T1 + b
    return line
w, b  = np.polyfit(T1,D1,deg=1)
equation = 'y = ' + str(round(w,8)) + 'x' ' + ' + str(round(b,3))
line = give_me_a_straight_line(T1,D1)
#ax.plot(T1,line,linestyle = '--', color = 'red')
#ax.text(0.8, 0.2,equation, horizontalalignment='center',
     #verticalalignment='center',
     #transform=ax.transAxes)


#ax.plot(T2, D2, color = 'black', marker = 'o', linestyle ='')


def give_me_a_straight_line(T2,D2):
    w, b  = np.polyfit(T2,D2,deg=1)
    line  = w * T2 + b
    return line
w, b  = np.polyfit(T2,D2,deg=1)
equation = 'y = ' + str(round(w,8)) + 'x' ' + ' + str(round(b,3))
line = give_me_a_straight_line(T2,D2)
#ax.plot(T2,line,linestyle = '--', color = 'blue')
#ax.text(0.8, 0.3,equation, horizontalalignment='center',
     #verticalalignment='center',
     #transform=ax.transAxes)

ax.plot(T3, D3, color = 'black', marker = 'o', linestyle ='')


def give_me_a_straight_line(T3,D3):
    w, b  = np.polyfit(T3,D3,deg=1)
    line  = w * T3 + b
    return line
w, b  = np.polyfit(T3,D3,deg=1)
equation = 'y = ' + str(336.2) + 'x' ' + ' + str(round(b,9))
line = give_me_a_straight_line(T3,D3)
ax.plot(T3,line,linestyle = '--', color = 'black')
ax.text(0.8, 0.4,equation, horizontalalignment='center',
     verticalalignment='center',
     transform=ax.transAxes)

#ax.plot(T4, D4, color = 'black', marker = 'o', linestyle ='')

def give_me_a_straight_line(T4,D4):
    w, b  = np.polyfit(T4,D4,deg=1)
    line  = w * T4 + b
    return line
w, b  = np.polyfit(T4,D4,deg=1)
equation = 'y = ' + str(round(w,8)) + 'x' ' + ' + str(round(b,3))
line = give_me_a_straight_line(T4,D4)
#ax.plot(T4,line,linestyle = '--', color = 'black')
#ax.text(0.8, 0.5,equation, horizontalalignment='center',
     #verticalalignment='center',
     #transform=ax.transAxes)






ax.set_xlabel('$t^{1.5}$ / s$^{1.5}$ ', fontsize = 16)
ax.set_ylabel('$t_{p}$ / s', fontsize = 16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)











