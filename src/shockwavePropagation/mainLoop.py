import numpy as np
from copy import deepcopy as cp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import scipy.sparse as sp
import math
import os
import sys
import tracemalloc

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

def save_stats(stats, name):
    f = open(name, 'w')
    for i in stats:
        if i.size / 1024 < 50:
            continue
        text = i.traceback[0].filename
        # text = text.split('\\')
        # f.write(text[0] + ' ' + text[len(text)-2] + '\\' + text[len(text)-1] + '  ')
        f.write(text)
        f.write(": %s memory blocks: %.1f KiB" % (i.count, i.size / 1024))
        f.write('\n')
    f.close()

def plot(n, p, u, T, rho, r, V, Vm, step, path):
    fig, ax = plt.subplots(5,1,figsize=(30,50))
    fig.tight_layout(pad=15.0)
    fig.subplots_adjust(top=0.94)
    x = np.array([x for x in range(int(PR.Params.L/PR.Params.dx)+1)])
    x = x * PR.Params.dx
    
    ax[0].plot(x, u.T)
    ax[0].axvline(r, ls = 'dashed', color = 'black', alpha = 0.5)
    ax[0].set_title("Velocity distribution", fontsize=40)
    ax[0].set_xlabel("distance [m]", fontsize=30)
    ax[0].tick_params(axis='x', labelsize=20)
    ax[0].set_ylabel("velocity [m/s]", fontsize=30)
    ax[0].tick_params(axis='y', labelsize=20)
    
    ax[1].plot(x, p.T*10**(-3))
    ax[1].axvline(r, ls = 'dashed', color = 'black', alpha = 0.5)
    ax[1].set_title("Pressure distribution", fontsize=40)
    ax[1].set_xlabel("distance [m]", fontsize=30)
    ax[1].tick_params(axis='x', labelsize=20)
    ax[1].set_ylabel("pressure [kPa]", fontsize=30)
    ax[1].tick_params(axis='y', labelsize=20)
    
    ax[2].plot(x, rho.T)
    ax[2].axvline(r, ls = 'dashed', color = 'black', alpha = 0.5)
    ax[2].set_title("Density distribution", fontsize=40)
    ax[2].set_xlabel("distance [m]", fontsize=30)
    ax[2].tick_params(axis='x', labelsize=20)
    ax[2].set_ylabel("density [kg/m3]", fontsize=30)
    ax[2].tick_params(axis='y', labelsize=20)
    
    ax[3].plot(x, n[[0],:].T)
    ax[3].plot(x, n[[1],:].T)
    ax[3].plot(x, n[[2],:].T)
    ax[3].plot(x, n[[3],:].T)
    ax[3].axvline(r, ls = 'dashed', color = 'black', alpha = 0.5)
    ax[3].set_title("Molar distribution", fontsize=40)
    ax[3].set_xlabel("distance [m]", fontsize=30)
    ax[3].tick_params(axis='x', labelsize=20)
    ax[3].set_ylabel("molar number [mol]", fontsize=30)
    ax[3].tick_params(axis='y', labelsize=20)
    
    ax[4].plot(x, T.T)
    ax[4].axvline(r, ls = 'dashed', color = 'black', alpha = 0.5)
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
    # linespace
    xx = range(int(PR.Params.L/PR.Params.dx)+1)
    x = np.zeros((1,len(xx)))
    x[0,:] = np.array([a for a in xx]) * PR.Params.dx
    
    u = np.ones((1,len(x[0,:])))
    u[0,x[0,:] <= PR.Params.r0] = PR.Params._u_D # m/s
    
    n = np.zeros((5, len(x[0,:])))
    n[3,:] = PR.Params.n_o2
    n[4,:] = PR.Params.n_n2
    
    rho = np.ones((1,len(x[0,:]))) * PR.Params._rho_air # kg/m3
    # rho[0,x[0,:] <= r0] = _rho0
    n[:,x[0,:] <= PR.Params.r0] = PR.Params.n_tnt * np.array([[0],[7],[5/2],[0],[3/2]])
    rho = rho_diff(n, rho, PR.Params.dx)
    Vmt = np.zeros((1,len(x[0,:])))
    Vmt[0,:] = Vm_calc(n, PR.Params.dx)
    
    p = np.ones((1,len(x[0,:]))) * PR.Params._p_ambient # Pa
    p[0,:] = p_diff(pvm, Vmt, PR.Params.gamma)
    
    T = np.zeros((1,len[x[0,:]]))
    # state = {'n': sp.csc_array(n), 'p': sp.csc_array(p), 'u': sp.csc_array(u), 'T': sp.csc_array(T), 'rho': sp.csc_array(rho), 'r': cp(0), 'V': sp.csc_array(V), 'Vm': sp.csc_array(Vm)}
    return n, p, u, T, rho, Vmt


def mainLoop():
    path = "E:/Simulations/Shockwave/"
    name = "Test5/"
    n, p, u, T, rho, r, V, Vm = init()
    step = 0
    plot(n, p, u, T, rho, r, V, Vm, step, path + name)
    l_idx = int(r * (int(1/PR.Params.dx)))
    
    # tracemalloc.start()
    while 1:
        # s = tracemalloc.take_snapshot()
        # save_stats(s.statistics('filename', cumulative=True), path+name+'stats_%i.txt' % step)
        # del s
        
        if step == int(PR.Params.Time/PR.Params.dt):
            break
        else:
            step = step + 1
        
        # radius
        f_idx = int(r * (int(1/PR.Params.dx)))
        print(f_idx)
        # r = r + u[f_idx]*PR.Params.dt
        l_idx = (r * (int(1/PR.Params.dx)))
        
        # advection / substance movement
        dn, n = AD.a_RK4(n, u, rho, PR.Params.dt, PR.Params.dx, PR.Params.dx**2) # n, u, rho, dt, dx, A
        
        print(n[0:5,0:4])
        print()
        # molar volume
        Vm1 = cp(Vm)
        Vm[0:f_idx+1] = PR.molarVolume(V, n, f_idx+1)
        # state['Vm'] = PR.volume(state['r'], PR.Params.dx,'spherical') / sum(sum(state['n']))
        
        # volume 
        # state['V'] = state['Vm'] * np.sum(state['n'], axis=1)
        
        # density
        PR.density(PR.molarMass(n, ['TNT','CO2','N2','H2O','O2']), V, f_idx+1, rho)
        
        # temperature
        # PR.temperature(dn, n, u, T, f_idx+1)
        del dn
        
        # RH condtion                
        # if f_idx < int(l_idx):
            # us, ps, dT = RH.jump(r, u, p, rho, PR.Params.dt, PR.Params.dx, f_idx)
            # print([us, ps, dT])
            # state['u',f_idx+1] = us
            # state['p',f_idx+1] = ps
            # f_idx = int(state['r'] * (int(1/PR.Params.dx)))

        # pressure        
        PR.pressure(Vm1, Vm, T, f_idx+1, p, n)
        del Vm1
        p[np.isinf(p)] = PR.Params.p_ambient
        p[np.isnan(p)] = PR.Params.p_ambient

        # velocity
        NS.u_RK4(u, PR.Params.dt, PR.Params.dx, p, rho, PR.Params.mu) # u, dt, dx, p, rho, mu
        
        if step % 100 == 0:
            plot(n, p, u, T, rho, r, V, Vm,step,path+name)

if __name__ == "__main__":
    mainLoop()