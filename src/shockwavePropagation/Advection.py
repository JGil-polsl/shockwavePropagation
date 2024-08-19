import numpy as np
import pandas as pd

from params import Sub_Molar_Mass
from params import Substance
from params import Params

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

def a_diff(n, u, rho, dt, dx, A):
    dn = np.zeros((len(n[:,1]), len(n[1,:])))
    # R = np.zeros((1,len(u[0,:])))
    # R[0,:] = A * rho * u * dt / 0.227 * (n[0,:] > 0)
    # R[0,:] = np.minimum(n[0,:], R)
    # R[0,:] = np.minimum(2/21 * n[3,:], R)
    # R[0,:] = np.minimum(n[0,:], n[3,:]*2/7)
    # s = np.array([[-1],[7],[5/2],[-7/2],[3/2]])
    
    dn[:,1:len(u[0,:])-1] = \
        - u[0,1:len(u[0,:])-1] * (n[:,2:len(u[0,:])] - n[:,1:len(u[0,:])-1])**2 / (2*dx) \
        + u[0,0:len(u[0,:])-2] * (n[:,1:len(u[0,:])-1]- n[:,0:len(u[0,:])-2])**2 / (2*dx) #\
        # +np.matmul(s,R)[:,1:len(R[0,:])]       
    return dn


def a_RK4(n, u, rho, dt, dx, A):
    dn1 = a_diff(n, u, rho, dt, dx, A) * dt
    dn2 = a_diff(n + dn1/2, u, rho, dt, dx, A) * dt
    dn3 = a_diff(n + dn2/2, u, rho, dt, dx, A) * dt
    dn4 = a_diff(n + dn3, u, rho, dt, dx, A) * dt
    dn = 1/6 * (dn1 + 2*dn2 + 2*dn3 + dn4)
    return dn

def kConst():
    return 1

# def Advection_Simple(u, n, density, dx, dt, front_r, A=1):
#     '''
#     Parameters
#     ----------
#     u : vector of floats
#         velocity distribution along shockwave.
#     n : pd.DataFrame
#         matrix containing distribution of substances.
#     density : vector of floats
#         density distribution along shockwave.
#     dx : float/vector of floats
#         change in dimension.
#     dt : float
#         change in time.
#     front_r: integer
#         shock front position
#     A : float, optional
#         Area of flow. The default is 1.

#     Returns
#     -------
#     dn: matrix of floats
#         matrix consisting difference in substances molar due to reaction.
#     n + dn + un: matrix of floats
#         matrix consisting substances molar : dn - due to reaction, un - due to advection
#     '''
#     dn = np.zeros((len(n),len(n[0])))
#     un = np.zeros((len(n),len(n[0])))
    
#     k_CO = kConst()
#     k_CO2 = kConst()
    
#     # n[Substance['O2'].value,0] = Params._rho_air * Params.dx**3 * 0.21 / Sub_Molar_Mass['O2']
#     # n[Substance['N2'].value,0] = Params._rho_air * Params.dx**3 * 0.79 / Sub_Molar_Mass['N2']
#     # TODO : dodac tlen, zakladamy nadcisnienie 0, ilosc n2 tez uzupelnic o powietrze
#     # adwekcja zle dziala, oprzec o prawo zachowania !
    
#     R = A*u[0]*density[0]*dt/Sub_Molar_Mass['TNT'] * (n[Substance['TNT'].value,0]>0) * (n[Substance['O2'].value,0]>0)
    
#     # un[Substance['TNT'].value,0] = - n[Substance['TNT'].value,0]*(u[0])*dt/dx 
#     un[Substance['TNT'].value,0] = - u[0] * (n[Substance['TNT'].value,0]) * dt / dx                                 
#     dn[Substance['TNT'].value,0] = - R
    
#     un[Substance['CO2'].value,0] = - u[0] * (n[Substance['CO2'].value,0]) * dt / dx
#     dn[Substance['CO2'].value,0] = 7*R
    
#     un[Substance['O2'].value,0] = - u[0] * (n[Substance['O2'].value,0]) * dt / dx \
#         + Params.u_ambient * (Params._rho_air * Params.dx**3 * 0.21 / Sub_Molar_Mass['O2']) * dt / dx
#     dn[Substance['O2'].value,0] = -(21/4)*R

#     un[Substance['N2'].value,0] = - u[0] * (n[Substance['N2'].value,0]) * dt / dx \
#         + Params.u_ambient * (Params._rho_air * Params.dx**3 * 0.79 / Sub_Molar_Mass['N2']) * dt / dx
#     dn[Substance['N2'].value,0] = 3/2 * R
    
#     # n[Substance['O2'].value,0] = Params._rho_air * Params.dx**3 * 0.21 / Sub_Molar_Mass['O2']
#     # n[Substance['N2'].value,0] = Params._rho_air * Params.dx**3 * 0.79 / Sub_Molar_Mass['N2']
    
#     un[Substance['H2O'].value,0] = - u[0] * (n[Substance['H2O'].value,0]) * dt / dx
#     dn[Substance['H2O'].value,0] = 5/2 * R    
    
#     for x in range(1,front_r):
#         R = A*u[x]*density[x]*dt/Sub_Molar_Mass['TNT'] * (n[Substance['TNT'].value,x]>0) * (n[Substance['O2'].value,0]>0)
        
#         un[Substance['TNT'].value,x] = - u[x] * (n[Substance['TNT'].value,x]) * dt / dx \
#             + u[x-1] * (n[Substance['TNT'].value,x-1]) * dt / dx
#         dn[Substance['TNT'].value,x] = - R
        
#         un[Substance['CO2'].value,x] = - u[x] * (n[Substance['CO2'].value,x]) * dt / dx \
#             + u[x-1] * (n[Substance['CO2'].value,x-1]) * dt / dx
#         dn[Substance['CO2'].value,x] = 7*R

#         un[Substance['O2'].value,x] = - u[x] * (n[Substance['O2'].value,x]) * dt / dx \
#             + u[x-1] * (n[Substance['O2'].value,x-1]) * dt / dx
#         dn[Substance['O2'].value,x] = -(21/4)*R

#         un[Substance['N2'].value,x] = - u[x] * (n[Substance['N2'].value,x]) * dt / dx \
#             + u[x-1] * (n[Substance['N2'].value,x-1]) * dt / dx
#         dn[Substance['N2'].value,x] = 3/2*R
        
#         un[Substance['H2O'].value,x] = - u[x] * (n[Substance['H2O'].value,x]) * dt / dx \
#             + u[x-1] * (n[Substance['H2O'].value,x-1]) * dt / dx
#         dn[Substance['H2O'].value,x] = 5/2*R
    
#     x = front_r   
#     if x > 0:
#         R = A*u[x]*density[x]*dt/Sub_Molar_Mass['TNT'] * (n[Substance['TNT'].value,x]>0) * (n[Substance['O2'].value,0]>0)
        
#         un[Substance['TNT'].value,x] = u[x-1] * (n[Substance['TNT'].value,x-1]) * dt / dx
#         dn[Substance['TNT'].value,x] = - R
        
#         un[Substance['CO2'].value,x] = u[x-1] * (n[Substance['CO2'].value,x-1]) * dt / dx
#         dn[Substance['CO2'].value,x] = 7*R

#         un[Substance['O2'].value,x] = u[x-1] * (n[Substance['O2'].value,x-1]) * dt / dx
#         dn[Substance['O2'].value,x] = -(21/4)*R

#         un[Substance['N2'].value,x] = u[x-1] * (n[Substance['N2'].value,x-1]) * dt / dx
#         dn[Substance['N2'].value,x] = 3/2*R
        
#         un[Substance['H2O'].value,x] = u[x-1] * (n[Substance['H2O'].value,x-1]) * dt / dx
#         dn[Substance['H2O'].value,x] = 5/2*R
    
#     n = n+dn+un
#     n[n < 0] = 0
#     del un
#     return dn, n


def Advection_Simple(u, n, density, dx, dt, front_r, step, A=1):
    '''
    Parameters
    ----------
    u : vector of floats
        velocity distribution along shockwave.
    n : pd.DataFrame
        matrix containing distribution of substances.
    density : vector of floats
        density distribution along shockwave.
    dx : float/vector of floats
        change in dimension.
    dt : float
        change in time.
    front_r: integer
        shock front position
    A : float, optional
        Area of flow. The default is 1.

    Returns
    -------
    dn: matrix of floats
        matrix consisting difference in substances molar due to reaction.
    n + dn + un: matrix of floats
        matrix consisting substances molar : dn - due to reaction, un - due to advection
    '''
    dn = np.zeros((len(n),len(n[0])))
    un = np.zeros((len(n),len(n[0])))
    
    k_CO = kConst()
    k_CO2 = kConst()
   
    x = 0
    R = A*u[x]*density[x]*dt/Sub_Molar_Mass['TNT'] * (n[Substance['TNT'].value,x]>0) * (n[Substance['O2'].value,0]>0)
    Ox = Params._rho_air * Params.dx**3 * 0.21 / Sub_Molar_Mass['O2']
    Nx = Params._rho_air * Params.dx**3 * 0.79 / Sub_Molar_Mass['N2']
    
    un[Substance['TNT'].value,x] = \
        + max(u[x],0) * n[Substance['TNT'].value,x] * dt / dx \
        - max(-u[x],0) * n[Substance['TNT'].value,x+1] * dt / dx \
        - max(Params.u_ambient,0) * 0 * dt / dx \
        + max(-Params.u_ambient,0) * n[Substance['TNT'].value,x] * dt / dx
    dn[Substance['TNT'].value,x] = - R
    
    un[Substance['CO2'].value,x] = \
        + max(u[x],0) * n[Substance['CO2'].value,x] * dt / dx \
        - max(-u[x],0) * n[Substance['CO2'].value,x+1] * dt / dx \
        - max(Params.u_ambient,0) * 0 * dt / dx \
        + max(-Params.u_ambient,0) * n[Substance['CO2'].value,x] * dt / dx
    dn[Substance['CO2'].value,x] = 7*R
   
    un[Substance['O2'].value,x] = \
        + max(u[x],0) * n[Substance['O2'].value,x] * dt / dx \
        - max(-u[x],0) * n[Substance['O2'].value,x+1] * dt / dx \
        - max(Params.u_ambient,0) * Ox * dt / dx \
        + max(-Params.u_ambient,0) * n[Substance['O2'].value,x] * dt / dx
    dn[Substance['O2'].value,x] = -(21/4)*R
   
    un[Substance['N2'].value,x] = \
        + max(u[x],0) * n[Substance['N2'].value,x] * dt / dx \
        - max(-u[x],0) * n[Substance['N2'].value,x+1] * dt / dx \
        - max(Params.u_ambient,0) * Nx * dt / dx \
        + max(-Params.u_ambient,0) * n[Substance['N2'].value,x] * dt / dx
    dn[Substance['N2'].value,x] = 3/2*R
    
    un[Substance['H2O'].value,x] = \
        + max(u[x],0) * n[Substance['H2O'].value,x] * dt / dx \
        - max(-u[x],0) * n[Substance['H2O'].value,x+1] * dt / dx \
        - max(Params.u_ambient,0) * 0 * dt / dx \
        + max(-Params.u_ambient,0) * n[Substance['H2O'].value,x] * dt / dx
    dn[Substance['H2O'].value,x] = 5/2*R
    
    for x in range(1,front_r-1):
        R = A*u[x]*density[x]*dt/Sub_Molar_Mass['TNT'] * (n[Substance['TNT'].value,x]>0) * (n[Substance['O2'].value,0]>0)
        
        un[Substance['TNT'].value,x] = \
            + max(u[x],0) * n[Substance['TNT'].value,x] * dt / dx \
            - max(-u[x],0) * n[Substance['TNT'].value,x+1] * dt / dx \
            - max(u[x-1],0) * n[Substance['TNT'].value,x-1] * dt / dx \
            + max(-u[x-1],0) * n[Substance['TNT'].value,x] * dt / dx
        dn[Substance['TNT'].value,x] = - R
        
        un[Substance['CO2'].value,x] = \
            + max(u[x],0) * n[Substance['CO2'].value,x] * dt / dx \
            - max(-u[x],0) * n[Substance['CO2'].value,x+1] * dt / dx \
            - max(u[x-1],0) * n[Substance['CO2'].value,x-1] * dt / dx \
            + max(-u[x-1],0) * n[Substance['CO2'].value,x] * dt / dx
        dn[Substance['CO2'].value,x] = 7*R
   
        un[Substance['O2'].value,x] = \
            + max(u[x],0) * n[Substance['O2'].value,x] * dt / dx \
            - max(-u[x],0) * n[Substance['O2'].value,x+1] * dt / dx \
            - max(u[x-1],0) * n[Substance['O2'].value,x-1] * dt / dx \
            + max(-u[x-1],0) * n[Substance['O2'].value,x] * dt / dx
        dn[Substance['O2'].value,x] = -(21/4)*R
   
        un[Substance['N2'].value,x] = \
            + max(u[x],0) * n[Substance['N2'].value,x] * dt / dx \
            - max(-u[x],0) * n[Substance['N2'].value,x+1] * dt / dx \
            - max(u[x-1],0) * n[Substance['N2'].value,x-1] * dt / dx \
            + max(-u[x-1],0) * n[Substance['N2'].value,x] * dt / dx
        dn[Substance['N2'].value,x] = 3/2*R
        
        un[Substance['H2O'].value,x] = \
            + max(u[x],0) * n[Substance['H2O'].value,x] * dt / dx \
            - max(-u[x],0) * n[Substance['H2O'].value,x+1] * dt / dx \
            - max(u[x-1],0) * n[Substance['H2O'].value,x-1] * dt / dx \
            + max(-u[x-1],0) * n[Substance['H2O'].value,x] * dt / dx
        dn[Substance['H2O'].value,x] = 5/2*R

    n = n+dn+un
    n[n < 0] = 0
    del un
    return dn, n
