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

def kConst():
    return 1

def Advection_Simple(u, n, density, dx, dt, front_r, A=1):
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
    # dn = pd.DataFrame(dn)
    # dn.columns = n.columns
    n[n < 0] = 0
    
    k_CO = kConst()
    k_CO2 = kConst()
    
    # TODO : dodac tlen, zakladamy nadcisnienie 0, ilosc n2 tez uzupelnic o powietrze
    
    un[0][Substance['TNT'].value] = - n[0][Substance['TNT'].value]*(u[0])*dt/dx 
    dn[0][Substance['TNT'].value] = - A*u[0]*density[0]*dt/Sub_Molar_Mass['TNT'] * (n[0][Substance['TNT'].value]>0)
    
    un[0][Substance['CO2'].value] = - n[0][Substance['CO2'].value]*(u[0])*dt/dx 
    dn[0][Substance['CO2'].value] = 7*A*u[0]*density[0]*dt/Sub_Molar_Mass['TNT'] * (n[0][Substance['TNT'].value]>0) 
    
    un[0][Substance['O2'].value] = - n[0][Substance['O2'].value]*(u[0])*dt/dx \
        + Params.n_o2 * Params.u_ambient * dt/dx
    dn[0][Substance['O2'].value] = -(21/4)*A*u[0]*density[0]*dt/Sub_Molar_Mass['TNT'] * (n[0][Substance['TNT'].value]>0)
        # - k_CO * n[0]['C']*dt 
    # dn[0]['CO'] = \
    #     - n[0]['CO']*(u[0] - u_ambient)*dt/dx \
    #     + k_CO*n[0]['C']*dt \
    #     - k_CO2*n[0]['CO']*dt
    # dn[0]['CO2'] = \
    #     - n[0]['CO2']*(u[0]-u_ambient)*dt/dx \
    #     + k_CO2*n[0]['CO']*dt
    un[0][Substance['N2'].value] = - n[0][Substance['N2'].value]*(u[0])*dt/dx 
    dn[0][Substance['N2'].value] = 3/2 * A*u[0]*density[0]*dt/Sub_Molar_Mass['TNT'] * (n[0][Substance['TNT'].value]>0)
    
    un[0][Substance['H2O'].value] = - n[0][Substance['H2O'].value]*(u[0])*dt/dx
    dn[0][Substance['H2O'].value] = 5/2 * A*u[0]*density[0]*dt/Sub_Molar_Mass['TNT'] * (n[0][Substance['TNT'].value]>0)    
    
    for x in range(1,front_r):
        un[x][Substance['TNT'].value] = - n[x][Substance['TNT'].value]*(u[x])*dt/dx \
            + n[x-1][Substance['TNT'].value]*(u[x-1])*dt/dx 
        dn[x][Substance['TNT'].value] = - A*u[x]*density[x]*dt/Sub_Molar_Mass['TNT'] * (n[x][Substance['TNT'].value]>0)
        
        un[x][Substance['CO2'].value] = - n[x][Substance['CO2'].value]*(u[x])*dt/dx \
            + n[x-1][Substance['CO2'].value]*(u[x-1])*dt/dx 
        dn[x][Substance['CO2'].value] = 7*A*u[x]*density[x]*dt/Sub_Molar_Mass['TNT'] * (n[x][Substance['TNT'].value]>0) #\

        un[x][Substance['O2'].value] = - n[x][Substance['O2'].value]*(u[x])*dt/dx \
            + n[x-1][Substance['O2'].value]*(u[x-1])*dt/dx
        dn[x][Substance['O2'].value] = -(21/4)*A*u[x]*density[x]*dt/Sub_Molar_Mass['TNT'] * (n[x][Substance['TNT'].value]>0)

            # - k_CO * n[x]['C']*dt 
        # dn[x]['CO'] = \
        #     - n[x]['CO']*(u[x] - u[x-1])*dt/dx \
        #     + k_CO*n[x]['C']*dt \
        #     - k_CO2*n[x]['CO']*dt
        # dn[x]['CO2'] = \
        #     - n[x]['CO2']*(u[x] - u[x-1])*dt/dx \
        #     + k_CO2*n[x]['CO']*dt
        un[x][Substance['N2'].value] = - n[x][Substance['N2'].value]*(u[x])*dt/dx \
            + n[x-1][Substance['N2'].value]*(u[x-1])*dt/dx 
        dn[x][Substance['N2'].value] = 3/2 * A*u[x]*density[x]*dt/Sub_Molar_Mass['TNT'] * (n[x][Substance['TNT'].value]>0)
        
        un[x][Substance['H2O'].value] = - n[x][Substance['H2O'].value]*(u[x])*dt/dx \
            + n[x-1][Substance['H2O'].value]*(u[x-1])*dt/dx 
        dn[x][Substance['H2O'].value] = 5/2 * A*u[x]*density[x]*dt/Sub_Molar_Mass['TNT'] * (n[x][Substance['TNT'].value]>0)
    
    x = front_r
    un[x][Substance['TNT'].value] = n[x-1][Substance['TNT'].value]*(u[x-1])*dt/dx 
    dn[x][Substance['TNT'].value] = - A*u[x]*density[x]*dt/Sub_Molar_Mass['TNT'] * (n[x][Substance['TNT'].value]>0)
    
    un[x][Substance['CO2'].value] = n[x-1][Substance['CO2'].value]*(u[x-1])*dt/dx 
    dn[x][Substance['CO2'].value] = 7*A*u[x]*density[x]*dt/Sub_Molar_Mass['TNT'] * (n[x][Substance['TNT'].value]>0)
    
    un[x][Substance['O2'].value] = n[x-1][Substance['O2'].value]*(u[x-1])*dt/dx
    dn[x][Substance['O2'].value] = -(21/4)*A*u[x]*density[x]*dt/Sub_Molar_Mass['TNT'] * (n[x][Substance['TNT'].value]>0)

        # - k_CO * n[x]['C']*dt 
    # dn[x]['CO'] = \
    #     - n[x]['CO']*(u[x] - u[x-1])*dt/dx \
    #     + k_CO*n[x]['C']*dt \
    #     - k_CO2*n[x]['CO']*dt
    # dn[x]['CO2'] = \
    #     - n[x]['CO2']*(u[x] - u[x-1])*dt/dx \
    #     + k_CO2*n[x]['CO']*dt
    un[x][Substance['N2'].value] = n[x-1][Substance['N2'].value]*(u[x-1])*dt/dx
    dn[x][Substance['N2'].value] = 3/2 * A*u[x]*density[x]*dt/Sub_Molar_Mass['TNT'] * (n[x][Substance['TNT'].value]>0)
    
    un[x][Substance['H2O'].value] = n[x-1][Substance['H2O'].value]*(u[x-1])*dt/dx 
    dn[x][Substance['H2O'].value] = 5/2 * A*u[x]*density[x]*dt/Sub_Molar_Mass['TNT'] * (n[x][Substance['TNT'].value]>0)        
    
    return dn, n+dn+un

            