# -*- coding: utf-8 -*-
"""
Python script to run "monte-carlo-optimized" strontium box model

REVERSE WEATHERING 11/23, BASALT ALTERATION 12/23 MODERN VALUE (Li and West, 2014)
@author: Xi-Kai Wang, Datu Adiatma,  Xiao-Ming Liu
"""
# Import modules and libraries
# ----------------------------

# Import time to keep track of timing
from time import time

# Import the trifecta of python scientific computation module
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import matplotlib.pyplot as plt

# Import module to run monte carlo simulation
from LithiumStochastic_updated import *

#define monte carlo parameters
mc_parameter = {
    "tmin"     :  180,
    "tmax"     :  145,
    "nt"       :  700,

    "Fr"     :  [5e9, 30e9],
    "Rr"     :  [0, 20],

    "Fh"       :  [10e9,20e9],
    "Rh"       :  [8.3, 8.3],

    "Dsed"       :  [0, 20],

    "Dalt"       :  [0, 15],

    "sampling" : 80000
}

scenario_1 = run_sim(mc_parameter, "target_d7Li")


#unpack results into variables
Fr = scenario_1['Fr']
Rr = scenario_1['Rr']
Fh = scenario_1['Fh']
Rh = scenario_1['Rh']
Fsed = scenario_1['Fsed']
Dsed = scenario_1['Dsed']
Falt = scenario_1['Falt']
Dalt = scenario_1['Dalt']
age = scenario_1['age']

#Array selector to choose non_zero values
non_zero = np.where(Fr!= 0)

#calculate seawater steady state solution
Rsw = np.zeros_like(Fr)
Rsw[non_zero] =(Fr[non_zero] * Rr[non_zero] + Fh[non_zero] * Rh[non_zero] 
                + Fsed[non_zero] * Dsed[non_zero] + Falt[non_zero] * Dalt[non_zero])/ (Fr[non_zero] + Fh[non_zero])


#Calculate mean and standard deviation of solution space
# create empty array to store solutions
#riverine
Fr_mean = np.zeros_like(age)
Fr_stdev = np.zeros_like(age)
Fr_max = np.zeros_like(age)
Fr_min = np.zeros_like(age)

Rr_mean = np.zeros_like(age)
Rr_stdev = np.zeros_like(age)
Rr_max = np.zeros_like(age)
Rr_min = np.zeros_like(age)

#hydrothermal 
Fh_mean = np.zeros_like(age)
Fh_stdev = np.zeros_like(age)
Fh_max = np.zeros_like(age)
Fh_min = np.zeros_like(age)

Rh_mean = np.zeros_like(age)
Rh_stdev = np.zeros_like(age)
Rh_max = np.zeros_like(age)
Rh_min = np.zeros_like(age)

#reverse weathering
Fsed_mean = np.zeros_like(age)
Fsed_stdev = np.zeros_like(age)
Fsed_max = np.zeros_like(age)
Fsed_min = np.zeros_like(age)

Dsed_mean = np.zeros_like(age)
Dsed_stdev = np.zeros_like(age)
Dsed_max = np.zeros_like(age)
Dsed_min = np.zeros_like(age)

#basalt alteration
Falt_mean = np.zeros_like(age)
Falt_stdev = np.zeros_like(age)
Falt_max = np.zeros_like(age)
Falt_min = np.zeros_like(age)

Dalt_mean = np.zeros_like(age)
Dalt_stdev = np.zeros_like(age)
Dalt_max = np.zeros_like(age)
Dalt_min = np.zeros_like(age)

#seawater
Rsw_mean = np.zeros_like(age)
Rsw_stdev = np.zeros_like(age)

for i in range(len(age)):
    Fr_d = Fr[i,:]
    Fr_mean[i] = np.mean(Fr_d[Fr_d!=0])
    Fr_stdev[i] = np.std(Fr_d[Fr_d!=0])

    Rr_d = Rr[i,:]
    Rr_mean[i] = np.mean(Rr_d[Rr_d!=0])
    Rr_stdev[i] = np.std(Rr_d[Rr_d!=0])

    Fh_d = Fh[i,:]
    Fh_mean[i] = np.mean(Fh_d[Fh_d!=0])
    Fh_stdev[i] = np.std(Fh_d[Fh_d!=0])

    Rh_d = Rh[i,:]
    Rh_mean[i] = np.mean(Rh_d[Rh_d!=0])
    Rh_stdev[i] = np.std(Rh_d[Rh_d!=0])

    Fsed_d = Fsed[i,:]
    Fsed_mean[i] = np.mean(Fsed_d[Fsed_d!=0])
    Fsed_stdev[i] = np.std(Fsed_d[Fsed_d!=0])

    Dsed_d = Dsed[i,:]
    Dsed_mean[i] = np.mean(Dsed_d[Dsed_d!=0])
    Dsed_stdev[i] = np.std(Dsed_d[Dsed_d!=0])

    Falt_d = Falt[i,:]
    Falt_mean[i] = np.mean(Falt_d[Falt_d!=0])
    Falt_stdev[i] = np.std(Falt_d[Falt_d!=0])

    Dalt_d = Dalt[i,:]
    Dalt_mean[i] = np.mean(Dalt_d[Dalt_d!=0])
    Dalt_stdev[i] = np.std(Dalt_d[Dalt_d!=0])

    Rsw_d = Rsw[i,:]
    Rsw_mean[i] = np.mean(Rsw_d[Rsw_d!=0])
    Rsw_stdev[i] = np.std(Rsw_d[Rsw_d!=0])

#---------------------------------------------------------------
#Error band
Fr_hi = Fr_mean + Fr_stdev
Fr_lo = Fr_mean - Fr_stdev
Rr_hi = Rr_mean + Rr_stdev
Rr_lo = Rr_mean - Rr_stdev

Fh_hi = Fh_mean + Fh_stdev
Fh_lo = Fh_mean - Fh_stdev
Rh_hi = Rh_mean + Rh_stdev
Rh_lo = Rh_mean - Rh_stdev

Fsed_hi = Fsed_mean + Fsed_stdev
Fsed_lo = Fsed_mean - Fsed_stdev
Dsed_hi = Dsed_mean + Dsed_stdev
Dsed_lo = Dsed_mean - Dsed_stdev

Falt_hi = Falt_mean + Falt_stdev
Falt_lo = Falt_mean - Falt_stdev
Dalt_hi = Dalt_mean + Dalt_stdev
Dalt_lo = Dalt_mean - Dalt_stdev

Rsw_hi = Rsw_mean + Rsw_stdev
Rsw_lo = Rsw_mean - Rsw_stdev



#---------------------------------------------------------------   
#Run transient model

nt = len(age)
dt =(age.max() - age.min())*1e6/nt
n  = np.ones(nt) * 3.4e16

Rsw_transient = np.zeros(nt)
Rsw_transient_hi = np.zeros(nt)
Rsw_transient_lo = np.zeros(nt)

Rsw_transient =run_Li(n,nt,dt,age,Rsw_transient,Fr_mean, Rr_mean, Fh_mean, Rh_mean, Fsed_mean, Dsed_mean, Falt_mean, Dalt_mean)

#Print out mean values
print("Fr = ",Fr_mean[0])
print("Rr = ",Rr_mean[0])
print("Fh = ",Fh_mean[0])
print("Rh = ",Rh_mean[0])
print("Fsed = ",Fsed_mean[0])
print("Dsed = ",Dsed_mean[0])
print("Falt = ",Falt_mean[0])
print("Dalt = ",Dalt_mean[0])
print(Rsw_transient[0])
print(Fr_mean[0] + Fh_mean[0]-Fsed_mean[0]-Falt_mean[0])
# #---------------------------------------------------------------  
#get the non-zero data at 180 Ma
non_zero = np.where(Fr[0,:] != 0)
Fr_initial = Fr[0,:][non_zero]
Fh_initial = Fh[0,:][non_zero]
Rr_initial = Rr[0,:][non_zero]
Rh_initial = Rh[0,:][non_zero]
Dalt_initial = Dalt[0,:][non_zero]
Dsed_initial = Dsed[0,:][non_zero]


#change the plot font
plt.rcParams['font.family'] = 'serif'  # Specifies the font family for text
plt.rcParams['font.serif'] = 'Times New Roman'  # Specific font within the family
plt.rcParams['font.size'] = 20  # Global font size


#Draw the figures for the monte carolo results
fig, ax = plt.subplots(3, 1, figsize = (10, 25))

#Figure 1 riverine flux and riverine ratio
ax[0].scatter(Fr_initial,Rr_initial, color = "grey", s = 1)
ax[0].scatter(Fr_mean[0],Rr_mean[0], color = "blue", s = 200, marker = "s")
ax[0].set_xlim(5e9, 30e9)
ax[0].set_ylim(0, 20)
ax[0].set_xlabel("FRiv (Gmol/y)")
ax[0].set_ylabel("RRiv (d7Li)")

#Figure 2 riverine flux and hydrothermal flux
ax[1].scatter(Fr_initial,Fh_initial, color = "grey", s = 1)
ax[1].scatter(Fr_mean[0],Fh_mean[0], color = "blue", s = 200, marker = "s")
ax[1].set_xlim(5e9, 30e9)
ax[1].set_ylim(10e9, 20e9)
ax[1].set_xlabel("FRiv (Gmol/y)")
ax[1].set_ylabel("Fh (Gmol/y)")

#Figure 3 basalt alteration and reverse weathering isotope ratio
ax[2].scatter(Dalt_initial,Dsed_initial, color = "grey", s = 1)
ax[2].scatter(Dalt_mean[0],Dsed_mean[0], color = "blue", s = 200, marker = "s")
ax[2].set_xlim(0, 15)
ax[2].set_ylim(0, 20)
ax[2].set_xlabel("Dalt (d7Li)")
ax[2].set_ylabel("Dsed (d7Li)")

plt.show()
# save figure
# fig.savefig('Sceario_1.eps', format='eps')

