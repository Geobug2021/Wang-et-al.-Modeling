
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess

"""
Python script Monte-Carlo box model

@author:  Xi-Kai Wang, Datu Adiatma, Xiao-Ming Liu
"""

# Steady state mass balance equation
def calc_RLisw_steady(Fr, Rr, Fh, Rh, Fsed, Dsed, Falt, Dalt):
    """Seawater d7Li at steady state
    
    Paramaters
    ----------
    Fr : float
        Riverine input flux
    Rr : float
        Riverine d7Li ratio
    Fh : float
        Hydrothermal input flux
    Rh : float
        Hydrothermal isotope ratio
    Fsed : float
        Lithium sink due to uptake onto marine sediments (Reverse Weathering)
    Dsed : float
        Lithium isotope fractionation due to reverse weathering
    Falt : float
        Lithium sink via basaltic alteration (oceanic crust)
    Dalt :float
        Lithium isotope fractionation due to basaltic alteration

    Return
    ------
    RLisw_ss : float
        Seawater lithium isotopic ratios at steady state
    """
    RLisw_ss = (Fr*Rr + Fh*Rh + Dsed*Fsed + Falt*Dalt) / (Fr + Fh)
    return RLisw_ss

# Function to run stochastic model
def run_sim(parameters, target_array, tolerance=0.6, mode='random', #0.6 is PMS xikai long term 2SD
            riverine_flux = [],
            riverine_ratio = [],
            riverine_age = [],
            hydrothermal_flux = [],
            hydrothermal_ratio = [],
            hydrothermal_age = [],
            reverseweathering_flux = [],
            rw_fractionation = [],
            rw_age = [],
            basaltalteration_flux = [],
            alt_fractionation = [],
            alt_age = []):
    """ Run stochastic modeling simulation

    Parameters
    ----------
    parameters : dict or json
        parameters to run the simulation
    target : dict or json
        d7Li values that are used as "target"
    tolerance : float, optional
        window of tolerance. Default is 1.1 permil
    mode : string
        Simulation mode. It can be:
            random : all parameters are randomized
            riverine : riverine flux are set to certain values
            riverine_ratios : riverine ratios are set to certain values
            hydrothermal :
            hydrothermal_ratios :
            reverse_weathering :
            etc
    
    Return
    ------
    results : dictionary
        Dictionary containing array of results    
    """

    # Load dictionaries or json files into variables
    if type(parameters) == dict:
        param = parameters
    else:
        with open(parameters) as f:
            param = json.load(f)
    
    if type(target_array) == dict:
        target = target_array
    else:
        with open(target_array) as f:
            target = json.load(f)
    
    # Unpack parameters into variables
    tmin = param['tmin']
    tmax = param['tmax']
    nt = param['nt']

    # Age in Ma
    age = np.linspace(tmin, tmax, nt)

    # Monte Carlo resampling
    s = param['sampling']

    """
    This monte carlo model allows filtering using lowess boundaries for lithium isotopes.

    """ 

    #LOWESS fitting to get the upper and lower bounds, calculate 200 times to save time, and do the resampling
    eval_x = np.linspace(tmin, tmax, 200)
    smoothed, bottom, top = lowess_with_confidence_bounds(
    target['age'], (target['d7Li']), eval_x, lowess_kw={"frac": 2/3}, N = 200
)

    #Resample target upper and lower boundary to fit the size of age array
    resample_bottom = interp1d(eval_x, bottom, bounds_error=False, fill_value='extrapolate' )
    bottom = resample_bottom(age)

    resample_top = interp1d(eval_x, top, bounds_error=False, fill_value='extrapolate' )
    top = resample_top(age)

    #Reshape bottom and top into nt * s matrix using zeros and transpose
    bottom = (np.zeros((s, nt)) + bottom).T
    top = (np.zeros((s, nt)) + top).T


    # Initiate random number generator with seed = 614 for reproducibility
    rng = np.random.default_rng(614)

    # Simulation with 'random' as mode
    if mode == 'random':
        print('random mode')
        Fr_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Fr'][0], param['Fr'][1], s))
        
        Rr_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Rr'][0], param['Rr'][1], s))
        
        Fh_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Fh'][0], param['Fh'][1], s))
        
        Rh_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Rh'][0], param['Rh'][1], s))
        
        Dalt_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Dalt'][0], param['Dalt'][1], s))

        Dsed_range = (np.zeros((nt, s)) +
                    rng.uniform(param['Dsed'][0], param['Dsed'][1], s))

        #Falt and Fsed are proportionate to the input， modern ratio
       
        Falt_range = (Fr_range + Fh_range) * (12/23) 
        Fsed_range = (Fr_range + Fh_range) * (11/23)
    
        # Empty list to store results
        Fr_res = np.zeros((nt, s))
        Rr_res = np.zeros((nt, s))
        Fh_res = np.zeros((nt, s))
        Rh_res = np.zeros((nt, s))
        Fsed_res = np.zeros((nt, s))
        Dsed_res = np.zeros((nt, s))
        Falt_res = np.zeros((nt, s))
        Dalt_res = np.zeros((nt, s))

        RLisw_ss = calc_RLisw_steady(Fr_range, Rr_range, Fh_range, 
                                     Rh_range, Fsed_range, Dsed_range,
                                     Falt_range, Dalt_range)

        # Filter
        print('Filtering Process')
 
        #new filtering boundary by Lowess FITTING
        Fr_res = np.where(((RLisw_ss > bottom) & (RLisw_ss < top)), Fr_range, 0)
        Rr_res = np.where(((RLisw_ss > bottom) & (RLisw_ss < top)), Rr_range, 0)
        Fh_res = np.where(((RLisw_ss > bottom) & (RLisw_ss < top)), Fh_range, 0)
        Rh_res = np.where(((RLisw_ss > bottom) & (RLisw_ss < top)), Rh_range, 0)
        Fsed_res = np.where(((RLisw_ss > bottom) & (RLisw_ss < top)), Fsed_range, 0)
        Dsed_res = np.where(((RLisw_ss > bottom) & (RLisw_ss < top)), Dsed_range, 0)
        Falt_res = np.where(((RLisw_ss > bottom) & (RLisw_ss < top)), Falt_range, 0)
        Dalt_res = np.where(((RLisw_ss > bottom) & (RLisw_ss < top)), Dalt_range, 0)

        sol = Fr_res[Fr_res!=0]

        print('solutions found', len(sol))

        # Store filtered results as a dict
        results ={
            'Fr' : Fr_res,
            'Rr' : Rr_res,
            'Fh' : Fh_res,
            'Rh' : Rh_res,
            'Fsed' : Fsed_res,
            'Dsed' : Dsed_res,
            'Falt' : Falt_res,
            'Dalt' : Dalt_res,
            'age' : age
        }

    return results
 


def simNLi(Fr, Fh, Fsed, Falt):
    """
    Lithium Mass Balance

    Parameters
    ----------
    Fr : float
        Global riverine flux of Li
    Fh : float
        Hydrothermal flux of Li
    Fsed : float
        reverse weathering  
    Fl: low-T alteration
    
    Returns
    -------
    nLi : float
        Seawater Li reservoir size in mol
    """
    nLi = Fr + Fh - Fsed - Falt
    return nLi

def simRLi(Dsw, N, Fr, Rr, Fh, Rh, Fsed, Dsed, Falt, Dalt):
    """
    N ：total amount of Li in seawater
    
    Dsw: seawater d7Li
    
    BOX MODEL INPUT FLUXES:
    Fr: riverine Li flux in Gmol/y
    Dr: riverine d7Li
    
    Fh: high-T hydrothermal Li flux in Gmol/y
    Dh: high-T hydrothermal d7Li
   
    BOX MODEL OUTPUT FLUXES:
    
    Falt: low-T hydrothermal Li flux in Gmol/y
    Dalt: fraction factor between Falt and seawater

    Fsed : maac Li flux in Gmol/y
    Dsed : fraction factor between maac and seawater

    """
    d7Li = ((Fr * (Rr - Dsw)) + Fh * (Rh - Dsw) + Fsed * Dsed + Falt * Dalt)/N
    
    return d7Li


# def run_Li (N, nt, dt, age_, rsw, Fr_, Rr_, Fh_, Rh_, Fsed_, Dsed_, Falt_, Dalt_):
def run_Li (N, nt, dt, age, rsw, fr, rr, fh, rh, fsed, dsed, falt, dalt):
    """
    solving diff. equations using forward Euler method

    Parameters:
    
    nt : numbers of time steps
    dt: the size of each time step
    age: age in million years

    BOX MODEL INPUT FLUXES:
    Fr: riverine Li flux in Gmol/y
    Dr: riverine d7Li
    
    Fh: high-T hydrothermal Li flux in Gmol/y
    Dh: high-T hydrothermal d7Li
   
    BOX MODEL OUTPUT FLUXES:
    
    Falt: low-T hydrothermal Li flux in Gmol/y
    Dalt: fraction factor between Falt and seawater

    Fsed : maac Li flux in Gmol/y
    Dsed : fraction factor between maac and seawater

    Returns:
    Dsw: lithium isotope ratio of seawter

    Assuming steady state at the beginning:
    Fsw[0] = (Jr[0] * Fr[0] + Jh[0] * Fh[0] - Jsed[0] * Fsed[0]) - Jl[0] * Fl[0]/(Jr[0] + Jh[0] - Jsed[0] - Jl[0])
  
    """

    #steady state:
    rsw0 = (fr[0]* rr[0] + rh[0]* fh[0] + dsed[0] * fsed[0] + falt[0]* dalt[0]) / (fr[0] + fh[0])
    rsw[0] = rsw0
    for i in range(nt-1):
        N[i+1] = N[i] + simNLi(fr[i], fh[i], fsed[i],falt[i])
        rsw[i+1]= rsw[i] + (simRLi(rsw[i], N[i], fr[i], rr[i], fh[i], rh[i], fsed[i], dsed[i], falt[i], dalt[i]) * dt)
        
    return rsw

#lowess confidence interval
def lowess_with_confidence_bounds(
    x, y, eval_x, N, conf_interval=0.95, lowess_kw=None
):
    """
    Perform Lowess regression and determine a confidence interval by bootstrap resampling
    """
    #convert x and y into array
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    if not isinstance(y, np.ndarray):
        y = np.array(y)

    # Lowess smoothing
    smoothed = lowess(exog=x, endog=y, xvals=eval_x, **lowess_kw)

    # Perform bootstrap resamplings of the data
    # and  evaluate the smoothing at a fixed set of points
    smoothed_values = np.empty((N, len(eval_x)))

    for i in range(N):
        sample = np.random.choice(len(x), len(x), replace=True)
        sampled_x = x[sample]
        sampled_y = y[sample]
        smoothed_values[i] = lowess(
            exog=sampled_x, endog=sampled_y, xvals=eval_x, **lowess_kw
        )

    # Get the confidence interval
    sorted_values = np.sort(smoothed_values, axis=0)
    bound = int(N * (1 - conf_interval) / 2)
    bottom = sorted_values[bound - 1]
    top = sorted_values[-bound]

    return smoothed, bottom, top
