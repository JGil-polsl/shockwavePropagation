import numpy as np
from enum import Enum
import math

'''
    Shockwaves propagation simulations basing on Gillespie algorithm.
    Copyright (C) 2022 by Jarosław Gil

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
def initial(mi, war, x):
    retVal = (mi*war)/((war/3) * math.sqrt(2*math.pi))
    retVal = retVal * np.exp((-1/2) * ((x)/(war/3))**2)
    return retVal

class Params():
    # global u_ambient, p_ambient, t_ambient, _rho_air
    dt = 1*10**(-9) # s
    dx = 10**(-3) # m
    Time = 10 # s
    L = 1 # m
    fi = 80 # mm
    mu = 1
    gamma = 1.4
    u_ambient = 1 # m/s
    t_ambient = 298 # K
    p_ambient = 101325 # Pa
    r0 = fi/2 * 10**(-3) # m
    _V_init = 4/3 * math.pi * r0**3 # m3
    m = 183.86 * 10**-3# kg
    _rho0 = m / _V_init #0.69 * 10**3 # g/cm3 -> 0.001 kg / 10^-6 m3
    _u_D = 6900 # m/s
    _T_D = 600 # K
    _rho_air = 1.293 # kg/m3 
    _p_init = 1 * 10**9 # Pa    
    n_o2 = _rho_air * dx**3 * 0.21 / 0.032

    
class Substance(Enum):
    TNT = 0
    C = 5
    CO = 6
    CO2 = 1
    N2 = 2
    H2O = 3
    O2 = 4
    

'''
Sub_Molar_Mass - dictionary to determine mass of compound [kg/mol]
    As index is interpreted chemical structure of compound ommiting sub and superscripts
'''
Sub_Molar_Mass = {
        "O2":   0.032,
        "N2":   0.028,
        "C":    0.012,
        "CO":   0.028,
        "CO2":  0.044,
        "H2O":  0.018,
        "TNT":  0.227,
        "air":  0.02897
    }

'''
Sub_Qr_re - dictionary to determine dH (kJ/mol) for chemical reactions [kJ/mol]
    As index is interpreted chemical structure of compound ommiting sub and superscripts
'''
Sub_Qr_re = {
        "O2":   0,
        "N2":   0,
        "C":    0,
        "CO":   0,
        "CO2":  393.13,
        "H2O":  245.18,
        "TNT":  -1036.028	
    }

R = 8.314
'''
Sub_C - dictionary to determine Cp for ideal gases [J/mol*K]
    As index is interpreted chemical structure of compound ommiting sub and superscripts
'''
Sub_C = {
    "Cp": {
        "1": 5/2 * R,
        "2": 7/2 * R,
        "3+": 8/2 * R
        },
    "Cv":{
        "1": 3/2 * R,
        "2": 5/2 * R,
        "3+": 6/2 * R
        },
    "TNT":  243.3,
    "CO2":  33.264,
    "N2":   29.092,
    "H2O":  33.228
    }

## parameter
def volume(r, dx, geo):
    '''
    Parameters
    ----------
    V : float
        explosive cloud volume in time t-1. [m3]
    dr : float or list of floats
        change in cloud dimensions. For spherical dr is parameter
        else dr should contain change in all dimensions [m]
    geo : text
        geometry acknowlodge. spherical, cubic, rectangural

    Returns
    -------
    float
        updated cloud volume. (V + dV) [m3]

    '''
    dV = 0
    if geo == "spherical":
        dV = 4/3 * math.pi * r**3
    return dV / math.floor((r*(int(1/dx)))+1)   
    
## vector
def density(Mmass, V, front_r, density):
    '''
    Parameters
    ----------
    Mmass : vector of floats
        mass distribution along shockwave. [kg]
    Vm : float
        molar volume. [m3/mol]
    n : matrix of floats
        molar number [mol]
    front_r : int
        front range
    density: vector of floats
        density distribution [kg/m3]

    Updates
    -------
    vector of floats
        density distribution along shockwave. [kg/m3]
    '''
    density[0:front_r] = Mmass[0:front_r]/(V[0:front_r])
 
## vector
def molarMass(n, ids):
    '''
    Parameters
    ----------
    n : vector of dicts
        mol distribution along shockwave set as dictionary.
        each cell of vector contains dictionary with keys as substance structure [mol]

    Returns
    -------
    _sum : vector of floats
        mass distribution along shockwave. [kg]
    '''
    global Sub_Molar_Mass
    n_sum = np.zeros(len(n[0,:]))
    for x in range(len(n[0,:])):
        for i in ids:
            n_sum[x] = n_sum[x] + n[Substance[i].value,x]*Sub_Molar_Mass[i]
    return n_sum
    
## vector
def temperature(dn, n, u, T, front_r):
    '''
    Parameters
    ----------
    dn : vector of dictionaries
        vector combined from differences in substances moles due to reaction. [mol]
    T : vector of floats
        actual temperature along shockwave. [K]

    Updates
    -------
    vector of floats
        new temperature as T + dT. [K]
    '''
    Qr = - abs(dn[Substance['TNT'].value,0])*Sub_Qr_re['TNT'] 
         # + abs(dn[0,Substance['CO2'].value])*Sub_Qr_re['CO2'] \
         # + abs(dn[0,Substance['H2O'].value])*Sub_Qr_re['H2O']             
        
    Cp = abs(n[Substance['N2'].value,0])*Sub_C['N2'] \
        + abs(n[Substance['CO2'].value,0])*Sub_C['CO2'] \
        + abs(n[Substance['H2O'].value,0])*Sub_C['H2O'] \
        + abs(n[Substance['TNT'].value][0])*Sub_C['TNT']
        # + n[i,Substance['O2'].value]*Sub_C['Cp']['2']
        # + n[i,Substance['CO'].value]*Sub_C['Cp']['2'] 
    T[0] = T[0] - Params.u_ambient * ( T[0] - Params.t_ambient ) * Params.dt / Params.dx \
          + Qr * 1000 * Params.dt / Cp
    
    for i in range(1, front_r):
        Qr = - abs(dn[Substance['TNT'].value,i])*Sub_Qr_re['TNT'] \
             + abs(dn[Substance['CO2'].value,i])*Sub_Qr_re['CO2'] \
             + abs(dn[Substance['H2O'].value,i])*Sub_Qr_re['H2O']             
            
        Cp = abs(n[Substance['N2'].value,i])*Sub_C['N2'] \
           + abs(n[Substance['CO2'].value,i])*Sub_C['CO2'] \
           + abs(n[Substance['H2O'].value,i])*Sub_C['H2O'] \
           + abs(n[Substance['TNT'].value,i])*Sub_C['TNT']
            # + n[i][Substance['O2'].value]*Sub_C['Cp']['2']
            # + n[i][Substance['CO'].value]*Sub_C['Cp']['2'] 
        T[i] = T[i] - u[i-1] * ( T[i] - T[i-1] ) * Params.dt / Params.dx \
             + Qr * 1000 / Cp
             
    # Qr = - abs(dn[i][Substance['TNT'].value])*Sub_Qr_re['TNT'] \
    #      + abs(dn[i][Substance['CO2'].value])*Sub_Qr_re['CO2'] \
    #      + abs(dn[i][Substance['H2O'].value])*Sub_Qr_re['H2O']             
        
    # Cp = abs(n[i][Substance['N2'].value])*Sub_C['N2'] \
    #     + abs(n[i][Substance['CO2'].value])*Sub_C['CO2'] \
    #     + abs(n[i][Substance['H2O'].value])*Sub_C['H2O'] \
    #     + abs(n[i][Substance['TNT'].value])*Sub_C['TNT']
    #     # + n[i][Substance['O2'].value]*Sub_C['Cp']['2']
    #     # + n[i][Substance['CO'].value]*Sub_C['Cp']['2'] 
    # T[i] = T[i] \
    #      + Qr * 1000 * Params.dt / Cp
             
## vector
def pressure(Vm1, Vm2, T, front_r, p, n):
    '''
    Parameters
    ----------
    Vm1 : vector of floats
        molar volume from t. [m3/mol]
    Vm2 : vector of floats
        molar volume from t+1. [m3/mol]
    T : vector of floats
        temperature distribution along shockwave. [K]
    front_r : int
        index of shock front
    p : vector of floats
        pressure distribution [Pa]
    n : matrix of floats
        molar distribution [mol]

    Updates
    -------
    p : vector of floats
        pressure distribution along shockwave. [Pa]
    '''
    # p = np.zeros(len(T))
    # p[0:front_r+1] = R * T[0:front_r+1] / (Vm)
    p[0:front_r] = p[0:front_r] * (Vm1[0:front_r]**(Params.gamma)) / (Vm2[0:front_r]**(Params.gamma))

## vector
def molarVolume(V, n, front_r):
    '''

    Parameters
    ----------
    V : float
        volume. [m3]
    n : float
        molar quantity. [mol]
    front_r : int
        shock front index

    Returns
    -------
    Vm : float
        molar volume. [m3/mol]

    '''
    if any(np.sum(n[1:5,:], axis = 0) == 0):
        print("wrong number")
    return V[0:front_r]/np.sum(n[1:5,:], axis=0)[0:front_r]