import numpy as np
import copy as cp
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
u_ambient = 0
p_ambient = 0

def NS_Euler(u, density, p, mu, dt, dx, front_r, step, f=0):
    '''
    Parameters
    ----------
    u : vector of floats
        velocity distribution along shockwave.
    density : vector of floats
        density distribution along shockwave.
    p : vector of floats
        pressure distribution along shockwave.
    mu : float
        dynamic viscosity of enviroment.
    dt : float
        time step.
    dx : float or vector of floats
        change in dimensions.
    f : float, optional
        additional force. The default is 9.81.
        
    Returns
    -------
    ut: vector of floats
        velocity distribution along shockwave in time t + dt as u + du.
    '''
    ## TODO do poprawy!!! cos zle liczy, za duza predkosc, nie maleje...
    ut = cp.copy(u)
    
    if step > 1:
        u[0] = Params.u_ambient
    
    for x in range(1, front_r-1):
        u[x] = ut[x] \
            - ut[x] * (ut[x] - ut[x-1]) * dt / dx\
            + f \
            - (1 / density[x]) * (p[x] - p[x-1]) * dt / dx \
            + mu * (ut[x+1] - 2*ut[x] + ut[x-1]) * dt / dx**2   
    del ut


def u_diff(u, dt, dx, p, rho, mu):
    du = np.zeros((1,len(u[0,:])))
    du[0,1:len(u[0,:])-1] = \
        - 0.5*u[0,1:len(u[0,:])-1] * u[0,1:len(u[0,:])-1] / dx \
        + 0.5*u[0,0:len(u[0,:])-2] * u[0,0:len(u[0,:])-2] / dx \
        + 4/3 *  mu * (u[0,2:len(u[0,:])] - 2*u[0,1:len(u[0,:])-1] + u[0,0:len(u[0,:])-2]) / dx**2 #\
        # - 1/rho[0,1:len(rho[0,:])-1] * p[0,1:len(p[0,:])-1] * dt / dx \
        # + 1/rho[0,0:len(rho[0,:])-2] * p[0,0:len(p[0,:])-2] * dt / dx#\
        # - 1/rho[0,1:len(rho[0,:])-1] * (p[0,1:len(p[0,:])-1] - p[0,0:len(p[0,:])-2]) * dt / dx
    return du

def u_RK4(u, dt, dx, p, rho, mu):
    du1 = u_diff(u, dt, dx, p, rho, mu) * dt
    du2 = u_diff(u + du1/2, dt, dx, p, rho, mu) * dt
    du3 = u_diff(u + du2/2, dt, dx, p, rho, mu) * dt
    du4 = u_diff(u + du3, dt, dx, p, rho, mu) * dt
    du = 1/6 * (du1 + 2*du2 + 2* du3 + du4)
    return du

def NS_diff():
    return 0

'''
old
    u[0] = ut[0] \
        - ut[0] * (ut[0] - Params.u_ambient) * dt / dx \
         + f \
        - (1 / density[0]) * (p[0] - p[1]) * dt / dx \
        + (1 / Params._rho_air) * (Params.p_ambient - p[0]) * dt / dx \
        + mu * (ut[1] - 2 * ut[0] + u_amb) * dt / dx**2
    
    for x in range(1, front_r):
        u[x] = ut[x] \
            - ut[x] * (ut[x] - ut[x-1]) * dt / dx \
            + f \
            - (1 / density[x]) * (p[x] - p[x+1]) * dt / dx \
            + (1 / density[x-1]) * (p[x-1] - p[x]) * dt / dx \
            + mu * (ut[x+1] - 2 * ut[x] + ut[x-1]) * dt / dx**2
    
    x = front_r
    if x > 0:
        u[x] = ut[x] \
            - ut[x] * (ut[x] - ut[x-1]) * dt / dx \
            + f \
            - (1 / density[x]) * (p[x] - p[x+1]) * dt / dx \
            + (1 / density[x-1]) * (p[x-1] - p[x]) * dt / dx \
            + mu * (ut[x+1] - 2 * ut[x] + ut[x-1]) * dt / dx**2
'''