import numpy as np
from copy import deepcopy as cp
import matplotlib.pyplot as plt
import math
import os

import params as PR
import Advection as AD
import NS 
import RH

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

def plot(state, step, path):
    fig, ax = plt.subplots(5,1,figsize=(30,50))
    fig.tight_layout(pad=15.0)
    fig.subplots_adjust(top=0.94)
    x = np.array([x for x in range(int(PR.Params.L/PR.Params.dx)+1)])
    x = x * PR.Params.dx
    
    ax[0].plot(x, state['u'])
    ax[0].set_title("Velocity distribution", fontsize=40)
    ax[0].set_xlabel("distance [m]", fontsize=30)
    ax[0].tick_params(axis='x', labelsize=20)
    ax[0].set_ylabel("velocity [m/s]", fontsize=30)
    ax[0].tick_params(axis='y', labelsize=20)
    
    ax[1].plot(x, state['p']*10**(-3))
    ax[1].set_title("Pressure distribution", fontsize=40)
    ax[1].set_xlabel("distance [m]", fontsize=30)
    ax[1].tick_params(axis='x', labelsize=20)
    ax[1].set_ylabel("pressure [kPa]", fontsize=30)
    ax[1].tick_params(axis='y', labelsize=20)
    
    ax[2].plot(x, state['rho'])
    ax[2].set_title("Density distribution", fontsize=40)
    ax[2].set_xlabel("distance [m]", fontsize=30)
    ax[2].tick_params(axis='x', labelsize=20)
    ax[2].set_ylabel("density [kg/m3]", fontsize=30)
    ax[2].tick_params(axis='y', labelsize=20)
    
    ax[3].plot(x, state['n'][:,0])
    ax[3].plot(x, state['n'][:,1])
    ax[3].plot(x, state['n'][:,2])
    ax[3].plot(x, state['n'][:,3])
    ax[3].set_title("Molar distribution", fontsize=40)
    ax[3].set_xlabel("distance [m]", fontsize=30)
    ax[3].tick_params(axis='x', labelsize=20)
    ax[3].set_ylabel("molar number [mol]", fontsize=30)
    ax[3].tick_params(axis='y', labelsize=20)
    
    ax[4].plot(x, state['T'])
    ax[4].set_title("Temperature distribution", fontsize=40)
    ax[4].set_xlabel("distance [m]", fontsize=30)
    ax[4].tick_params(axis='x', labelsize=20)
    ax[4].set_ylabel("temperature [K]", fontsize=30)
    ax[4].tick_params(axis='y', labelsize=20)
    
    fig.suptitle('Enviroment state, step: %i' % step, fontsize=75)
    
    try:
        os.makedirs(path + "Figures", exist_ok=True) 
    except OSError as error:
        print(error)
    finally:
        if os.path.exists(path + "Figures/%i.jpg" % step):
            os.remove(path + "Figures/%i.jpg" % step)
        fig.savefig(path + "Figures/%i.jpg" % step)
        plt.close(fig)          

def init():
    x = np.array(range(int(PR.Params.L/PR.Params.dx)+1))/(int(1/PR.Params.dx))
    nx = (PR.Params.m / PR.Sub_Molar_Mass['TNT'])
    Ox = PR.Params._rho_air * PR.Params.dx**3 * 0.21 / PR.Sub_Molar_Mass['O2']
    Nx = PR.Params._rho_air * PR.Params.dx**3 * 0.79 / PR.Sub_Molar_Mass['N2']
    V = np.ones(len(x)) * PR.Params.dx ** 3
    r = PR.Params.r0
    n = np.zeros((int(PR.Params.L/PR.Params.dx)+1, 5))
    n[0:int(r*(1/PR.Params.dx))+1,PR.Substance['TNT'].value] = PR.Params.dx ** 3 * nx/PR.Params._V_init    
    n[:,PR.Substance['N2'].value] = Nx
    n[:,PR.Substance['O2'].value] = Ox
    # Vmx = np.ones(len(x)) * (PR.Params.dx**3 / (PR.Params.dx**3 * PR.Params._rho_air / PR.Sub_Molar_Mass['air']))
    Vm = PR.molarVolume(V, n, len(x)+1)
    # Vm = V/n[:, PR.Substance['TNT'].value]
    # Vm[np.isinf(Vm)] = PR.Params.dx**3 / (PR.Params.dx**3 * PR.Params._rho_air / PR.Sub_Molar_Mass['air'])
    # V = Vm * n[:,PR.Substance['TNT'].value]
    # p = PR.initial(PR.Params._p_init, 3*r/5, x) + PR.Params.p_ambient
    p = np.ones(len(x)) * PR.Params.p_ambient
    # p = p * Vmx / Vm
    u = PR.initial(PR.Params._u_D, 3*r/5, x) + PR.Params.u_ambient
    # T = PR.initial(PR.Params._T_D, 3*r/5, x) + PR.Params.t_ambient
    T = np.ones(len(x)) * PR.Params.t_ambient
    # rho = n[:, PR.Substance['TNT'].value] * PR.Sub_Molar_Mass['TNT']/V + PR.Params._rho_air
    # rho = np.ones(len(x)) * PR.Params._rho_air
    # rho[np.isnan(rho)] = PR.Params._rho_air
    rho = np.zeros(len(x))
    PR.density(PR.molarMass(n, ['H2O', 'O2', 'N2', 'CO2']), V, len(x)+1, rho)
    # Vm = PR.Params.dx**3/n[:,0]
    # Vm[np.isinf(Vm)] = 0
    state = {'n': cp(n), 'p': cp(p), 'u': cp(u), 'T': cp(T), 'rho': cp(rho), 'r': cp(PR.Params.r0), 'V': cp(V), 'Vm': cp(Vm)}
    return state    


def mainLoop():
    path = "E:/Simulations/Shockwave/"
    name = "With O2/"
    state = init()
    step = 0
    plot(state, step, path + name)
    l_idx = int(state['r'] * (int(1/PR.Params.dx)))
    
    while 1:
        if step == int(PR.Params.Time/PR.Params.dt):
            break
        else:
            step = step + 1
        
        # radius
        f_idx = int(state['r'] * (int(1/PR.Params.dx)))
        print(f_idx)
        state['r'] = state['r'] + state['u'][f_idx]*PR.Params.dt
        l_idx = (state['r'] * (int(1/PR.Params.dx)))
        
        # advection / substance movement
        dn, state['n'] = AD.Advection_Simple(state['u'], state['n'], state['rho'], PR.Params.dx, PR.Params.dt, f_idx, A=PR.Params.dx**2)
        
        # molar volume
        Vm = cp(state['Vm'])
        state['Vm'][0:f_idx+1] = PR.molarVolume(state['V'], state['n'], f_idx+1)
        state['Vm'][np.isnan(state['Vm'])] = PR.Params.dx**3 / (PR.Params.dx**3 * PR.Params._rho_air / PR.Sub_Molar_Mass['air'])
        state['Vm'][np.isinf(state['Vm'])] = PR.Params.dx**3 / (PR.Params.dx**3 * PR.Params._rho_air / PR.Sub_Molar_Mass['air'])
        # state['Vm'] = PR.volume(state['r'], PR.Params.dx,'spherical') / sum(sum(state['n']))
        
        # volume 
        # state['V'] = state['Vm'] * np.sum(state['n'], axis=1)
        
        # density
        PR.density(PR.molarMass(state['n'], ['TNT','CO2','N2','H2O','O2']), state['V'], f_idx+1, state['rho'])
        
        # temperature
        PR.temperature(dn, state['n'], state['u'], state['T'], f_idx+1)
        
        # RH condtion                
        if f_idx < int(l_idx):
            us, ps, dT = RH.jump(state['r'], state['u'], state['p'], state['rho'], PR.Params.dt, PR.Params.dx, f_idx)
            print([us, ps, dT])
            # state['u'][f_idx+1] = us
            # state['p'][f_idx+1] = ps
            # f_idx = int(state['r'] * (int(1/PR.Params.dx)))

        # pressure        
        PR.pressure(Vm, state['Vm'], state['T'], f_idx+1, state['p'], state['n'])
        state['p'][np.isinf(state['p'])] = PR.Params.p_ambient
        state['p'][np.isnan(state['p'])] = PR.Params.p_ambient

        # velocity
        state['u'] = NS.NS_Euler(state['u'], state['rho'], state['p'], PR.Params.mu, PR.Params.dt, PR.Params.dx, f_idx)
        plot(state,step,path+name)

if __name__ == "__main__":
    mainLoop()