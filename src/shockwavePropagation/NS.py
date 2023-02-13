import numpy as np
import copy as cp
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

def NS_Euler(u, density, p, mu, dt, dx, front_r, f=9.81):
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
    nu = np.zeros(len(u))
    nu = cp.deepcopy(u)
    nu[0] = u[0] \
            - u[0]*(u[0])*dt/dx \
            + f*dt \
            - (p[0])/(density[0]) * dt/dx 
            # + mu*(u[1] - 2*u[0] + u_ambient)*dt/(dx**2)
    for i in range(1,front_r):
        nu[i] = u[i] \
                - u[i]*(u[i])*dt/dx \
                + u[i-1]*(u[i-1])*dt/dx \
                + f*dt \
                - (p[i])/(density[i]) * dt/dx \
                + (p[i-1])/(density[i-1]) * dt/dx 
   
    i = front_r
    nu[i] = u[i] \
            + u[i-1]*(u[i-1])*dt/dx \
            + f*dt \
            + (p[i-1])/(density[i-1]) * dt/dx 
                # + mu*(u[i+1] - 2*u[i] + u[i-1])*dt/(dx**2)
    # u[len(u)-1] = \
    #         - u[len(du)-1]*(u[len(du)-1]-u[len(du)-2])*dt/dx \
    #         + f*dt \
    #         + (p[len(du)-1] - p[len(du)-2])/density[len(du)-1] * dt/dx \
    #         + mu*(u_ambient - 2*u[len(du)-1] + u[len(du)-2])*dt/(dx**2)
    return nu

def NS_RK4(u, density, p, mu, dt, dx, front_r, f=9.81):
    nu = np.zeros(len(u))
    return nu

def NS_diff():
    return 0