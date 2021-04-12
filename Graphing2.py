# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 17:07:22 2021

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



data = pd.read_csv("QA complete data2.csv")
fig = plt.figure(figsize=(8,4),dpi=200)
ax=fig.add_subplot(111)
fig2 = plt.figure(figsize=(8,4),dpi=200)
ax2=fig2.add_subplot(111)





##############################################################1/V##################################################


CI = data.iloc[:,0]
CII = data.iloc[:,5]
CIII = data.iloc[:,10]
CIV = data.iloc[:,15]

CI = np.array(CI)
CII = np.array(CII)
CIII = np.array(CIII)
CIV = np.array(CIV)

CI = CI[~pd.isnull(CI)]
CII = CII[~pd.isnull(CII)]
CIII = CIII[~pd.isnull(CIII)]
CIV = CIV[~pd.isnull(CIV)]


###############################################################1/T####################################################


TCI = data.iloc[:,1]
TCII = data.iloc[:,6]
TCIII = data.iloc[:,11]
TCIV = data.iloc[:,16]

TCI = np.array(TCI)
TCII = np.array(TCII)
TCIII = np.array(TCIII)
TCIV = np.array(TCIV)

TCI = TCI[~pd.isnull(TCI)]
TCII = TCII[~pd.isnull(TCII)]
TCIII = TCIII[~pd.isnull(TCIII)]
TCIV = TCIV[~pd.isnull(TCIV)]


###############################################################T######################################################


TI = data.iloc[:,3]
TII = data.iloc[:,8]
TIII = data.iloc[:,13]
TIV = data.iloc[:,18]

TI = np.array(TI)
TII = np.array(TII)
TIII = np.array(TIII)
TIV = np.array(TIV)

TI = TI[~pd.isnull(TI)]
TII = TII[~pd.isnull(TII)]
TIII = TIII[~pd.isnull(TIII)]
TIV = TIV[~pd.isnull(TIV)]

#################################################################V####################################################

VI = data.iloc[:,2]
VII = data.iloc[:,7]
VIII = data.iloc[:,12]
VIV = data.iloc[:,17]

VI = np.array(VI)
VII = np.array(VII)
VIII = np.array(VIII)
VIV = np.array(VIV)

VI = VI[~pd.isnull(VI)]
VII = VII[~pd.isnull(VII)]
VIII = VIII[~pd.isnull(VIII)]
VIV = VIV[~pd.isnull(VIV)]


#################################################################PLOTTING############################################


ax.plot(TCI, VI, color = 'black', marker = 'o', linestyle ='')


def give_me_a_straight_line(TCI,VI):
    w, b  = np.polyfit(TCI,VI,deg=1)
    line  = w * TCI + b
    return line
w, b  = np.polyfit(TCI,VI,deg=1)
equation = 'y = ' + str(round(w,8)) + 'x' ' + ' + str(round(b,3))
line = give_me_a_straight_line(TCI,VI)
ax.plot(TCI,line,linestyle = '--', color = 'black')
ax.text(0.8, 0.2,equation, horizontalalignment='center',
     verticalalignment='center',
     transform=ax.transAxes)


ax.plot(TCII, VII, color = 'black', marker = 'o', linestyle ='')

def give_me_a_straight_line(TCII,VII):
    w, b  = np.polyfit(TCII,VII,deg=1)
    line  = w * TCII + b
    return line
w, b  = np.polyfit(TCII,VII,deg=1)
equation = 'y = ' + str(round(w,8)) + 'x' ' + ' + str(round(b,3))
line = give_me_a_straight_line(TCII,VII)
ax.plot(TCII,line,linestyle = '--', color = 'black')
ax.text(0.8, 0.3,equation, horizontalalignment='center',
     verticalalignment='center',
     transform=ax.transAxes)



ax.plot(TCIII, VIII, color = 'black', marker = 'o', linestyle ='')

def give_me_a_straight_line(TCIII,VIII):
    w, b  = np.polyfit(TCIII,VIII,deg=1)
    line  = w * TCIII + b
    return line
w, b  = np.polyfit(TCIII,VIII,deg=1)
equation = 'y = ' + str(round(w,8)) + 'x' ' + ' + str(round(b,3))
line = give_me_a_straight_line(TCIII,VIII)
ax.plot(TCIII,line,linestyle = '--', color = 'black')
ax.text(0.8, 0.4,equation, horizontalalignment='center',
     verticalalignment='center',
     transform=ax.transAxes)



ax.plot(TCIV, VIV, color = 'black', marker = 'o', linestyle ='')

def give_me_a_straight_line(TCIV,VIV):
    w, b  = np.polyfit(TCIV,VIV,deg=1)
    line  = w * TCIV + b
    return line
w, b  = np.polyfit(TCIV,VIV,deg=1)
equation = 'y = ' + str(round(w,7)) + 'x' ' + ' + str(round(b,3))
line = give_me_a_straight_line(TCIV,VIV)
ax.plot(TCIV,line,linestyle = '--', color = 'black')
ax.text(0.8, 0.5,equation, horizontalalignment='center',
     verticalalignment='center',
     transform=ax.transAxes)



ax.set_xlabel('Voltage / V', fontsize = 16)
ax.set_ylabel('Time$^{-1}$ / S$^{-1}$', fontsize = 16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.tick_params(axis='both', which='minor', labelsize=14)











ax2.plot(CI, TI, color = 'black', linestyle='', marker='o')

def give_me_a_straight_line(CI,TI):
    w, b  = np.polyfit(CI,TI,deg=1)
    line  = w * CI + b
    return line
w, b  = np.polyfit(CI,TI,deg=1)
equation = 'y = ' + str(round(w,8)) + 'x' ' + ' + str(round(b,9))
line = give_me_a_straight_line(CI,TI)
ax2.plot(CI,line,linestyle = '--', color = 'black')
ax2.text(0.15, 0.7,equation, horizontalalignment='center',
     verticalalignment='center',
     transform=ax.transAxes)



ax2.plot(CII, TII, color = 'black',linestyle='', marker='o')

def give_me_a_straight_line(CII,TII):
    w, b  = np.polyfit(CII,TII,deg=1)
    line  = w * CII + b
    return line
w, b  = np.polyfit(CII,TII,deg=1)
equation = 'y = ' + str(round(w,8)) + 'x' ' + ' + str(round(b,9))
line = give_me_a_straight_line(CII,TII)
ax2.plot(CII,line,linestyle = '--', color = 'black')
ax2.text(0.15, 0.8,equation, horizontalalignment='center',
     verticalalignment='center',
     transform=ax.transAxes)



ax2.plot(CIII, TIII, color = 'black',linestyle='', marker='o')

def give_me_a_straight_line(CIII,TIII):
    w, b  = np.polyfit(CIII,TIII,deg=1)
    line  = w * CIII + b
    return line
w, b  = np.polyfit(CIII,TIII,deg=1)
equation = 'y = ' + str(round(w,8)) + 'x' ' + ' + str(round(b,9))
line = give_me_a_straight_line(CIII,TIII)
ax2.plot(CIII,line,linestyle = '--', color = 'black')
ax2.text(0.15, 0.9,equation, horizontalalignment='center',
     verticalalignment='center',
     transform=ax.transAxes)



ax2.plot(CIV, TIV, color = 'black',linestyle='', marker='o')

def give_me_a_straight_line(CIV,TIV):
    w, b  = np.polyfit(CIV,TIV,deg=1)
    line  = w * CIV + b
    return line
w, b  = np.polyfit(CIV,TIV,deg=1)
equation = 'y = ' + str(round(w,7)) + 'x' ' + ' + str(round(b,9))
line = give_me_a_straight_line(CIV,TIV)
ax2.plot(CIV,line,linestyle = '--', color = 'black')
ax2.text(0.15, 1.0,equation, horizontalalignment='center',
     verticalalignment='center',
     transform=ax.transAxes)



ax2.set_ylabel('Time / s', fontsize = 16)
ax2.set_xlabel('Voltage$^{-1}$ / V$^{-1}$', fontsize = 16)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='minor', labelsize=14)





