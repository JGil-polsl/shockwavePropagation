import numpy as np
import scipy.sparse as sp
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import copy as cp
import os

def plot(x, n, p, u, rho, Vmt, step, dt, path):
    fig, ax = plt.subplots(4,1,figsize=(30,50))
    fig.tight_layout(pad=15.0)
    fig.subplots_adjust(top=0.94)
    
    ax[0].plot(x[0,:], u[0,:])
    # ax[0].axvline(r, ls = 'dashed', color = 'black', alpha = 0.5)
    ax[0].set_title("Velocity distribution", fontsize=40)
    ax[0].set_xlabel("distance [m]", fontsize=30)
    ax[0].tick_params(axis='x', labelsize=20)
    ax[0].set_ylabel("velocity [m/s]", fontsize=30)
    ax[0].tick_params(axis='y', labelsize=20)
    
    ax[1].plot(x[0,:], p[0,:]*10**(-3))
    # ax[1].axvline(r, ls = 'dashed', color = 'black', alpha = 0.5)
    ax[1].set_title("Pressure distribution", fontsize=40)
    ax[1].set_xlabel("distance [m]", fontsize=30)
    ax[1].tick_params(axis='x', labelsize=20)
    ax[1].set_ylabel("pressure [kPa]", fontsize=30)
    ax[1].tick_params(axis='y', labelsize=20)
    
    ax[2].plot(x[0,:], rho[0,:])
    # ax[2].axvline(r, ls = 'dashed', color = 'black', alpha = 0.5)
    ax[2].set_title("Density distribution", fontsize=40)
    ax[2].set_xlabel("distance [m]", fontsize=30)
    ax[2].tick_params(axis='x', labelsize=20)
    ax[2].set_ylabel("density [kg/m3]", fontsize=30)
    ax[2].tick_params(axis='y', labelsize=20)
    
    ax[3].plot(x[0,:], n[0,:])
    ax[3].plot(x[0,:], n[1,:])
    ax[3].plot(x[0,:], n[2,:])
    ax[3].plot(x[0,:], n[3,:])
    ax[3].plot(x[0,:], n[4,:])
    ax[3].plot(x[0,:], n[5,:])
    ax[3].plot(x[0,:], n[6,:])
    # ax[3].axvline(r, ls = 'dashed', color = 'black', alpha = 0.5)
    ax[3].set_title("Molar distribution", fontsize=40)
    ax[3].set_xlabel("distance [m]", fontsize=30)
    ax[3].tick_params(axis='x', labelsize=20)
    ax[3].set_ylabel("molar number [mol]", fontsize=30)
    ax[3].tick_params(axis='y', labelsize=20)
    
    fig.suptitle('Enviroment state, time: %.3f ms' % (step*1000*dt), fontsize=75)
    
    try:
        os.makedirs(path + "Figures", exist_ok=True) 
    except OSError as error:
        print(error)
    finally:
        if os.path.exists(path + "Figures/%i.jpg" % step):
            os.remove(path + "Figures/%i.jpg" % step)
        fig.savefig(path + "Figures/%i.jpg" % step)
        plt.close(fig)      

def save_data(x, n, u, p, rho, Vm, path, step):
    try:
        os.makedirs(path + 'composition', exist_ok=True) 
        os.makedirs(path + 'velocity', exist_ok=True) 
        os.makedirs(path + 'pressure', exist_ok=True) 
        os.makedirs(path + 'density', exist_ok=True) 
        os.makedirs(path + 'molar volume', exist_ok=True) 
    except OSError as error:
        print(error)
    finally:
        sp.save_npz(path + 'composition/n_' + str(step), sp.csr_matrix(n))
        sp.save_npz(path + 'velocity/u_' + str(step), sp.csr_matrix(u))
        sp.save_npz(path + 'pressure/p_' + str(step), sp.csr_matrix(p))
        sp.save_npz(path + 'density/rho_' + str(step), sp.csr_matrix(rho))
        sp.save_npz(path + 'molar volume/Vm_' + str(step), sp.csr_matrix(Vm))

def u_diff(u, dt, dx, p, rho):
    du = np.zeros((1,len(u[0,:])))
    du[0,1:len(u[0,:])-1] = \
        - 0.5*u[0,1:len(u[0,:])-1] * u[0,1:len(u[0,:])-1] / dx \
        + 0.5*u[0,0:len(u[0,:])-2] * u[0,0:len(u[0,:])-2] / dx \
        + 4/3 *  mu * (u[0,2:len(u[0,:])] - 2*u[0,1:len(u[0,:])-1] + u[0,0:len(u[0,:])-2]) / dx**2 #\
        # - 1/rho[0,1:len(rho[0,:])-1] * p[0,1:len(p[0,:])-1] * dt / dx \
        # + 1/rho[0,0:len(rho[0,:])-2] * p[0,0:len(p[0,:])-2] * dt / dx#\
        # - 1/rho[0,1:len(rho[0,:])-1] * (p[0,1:len(p[0,:])-1] - p[0,0:len(p[0,:])-2]) * dt / dx
    return du

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
    
def rho_diff(n, rho, dx):
    drho = np.zeros((1,len(rho[0,:])))
    drho[0,:] = np.matmul([0, 0.044, 0.028, 0, 0.018, 0.032, 0.028],n)/dx**3 # change 0.227 -> 0, change 0.044 -> 0.028
    return drho

def p_diff(pwm, Vmt, gamma):
    return pwm / Vmt**gamma

def Vm_calc(n, dx):
    return dx**3 / np.matmul([0, 1, 1, 0, 1, 1, 1],n) # change n -> n[1:5,:]

def u_RK4(u, dt, dx, p, rho):
    du1 = u_diff(u, dt, dx, p, rho) * dt
    du2 = u_diff(u + du1/2, dt, dx, p, rho) * dt
    du3 = u_diff(u + du2/2, dt, dx, p, rho) * dt
    du4 = u_diff(u + du3, dt, dx, p, rho) * dt
    du = 1/6 * (du1 + 2*du2 + 2* du3 + du4)
    return u + du

def u_Eul(u, dt, dx, p, rho, f_r): # TODO conservation at du/dx
    du = np.zeros((1,len(u[0,:])))
    du[0,0] = \
        - u[0,0] * (u[0,0] - 1) * dt / dx \
        + mu * (u[0,1] - 2*u[0,0] + 1) * dt / dx**2 \
        - (1/rho[0,1] - 1/rho[0,0]) * (p[0,1] - p[0,0]) * dt / dx
    du[0,1:len(u[0,:])-1] = \
        - u[0,1:len(u[0,:])-1] * (u[0,1:len(u[0,:])-1] - u[0,0:len(u[0,:])-2]) * dt / dx \
        + mu * (u[0,2:len(u[0,:])] - 2*u[0,1:len(u[0,:])-1] + u[0,0:len(u[0,:])-2]) * dt / dx**2 \
        - (1/rho[0,2:len(rho[0,:])] - 1/rho[0,1:len(rho[0,:])-1]) * (p[0,2:len(p[0,:])] - p[0,1:len(p[0,:])-1]) * dt / dx
    du[0,len(u[0,:])-1] = \
        - u[0,len(u[0,:])-1] * (u[0,len(u[0,:])-1] - u[0,len(u[0,:])-2]) * dt / dx \
        + mu * (1 - 2 * u[0,len(u[0,:])-1] + u[0,len(u[0,:])-2]) * dt / dx \
        - (1/_rho_air - 1/rho[0,len(rho[0,:])-1])* (_p_ambient - p[0,len(p[0,:])-1]) * dt / dx                                           
    return u + du

def a_RK4(n, u, rho, dt, dx, A):
    dn1 = a_diff(n, u, rho, dt, dx, A) * dt
    dn2 = a_diff(n + dn1/2, u, rho, dt, dx, A) * dt
    dn3 = a_diff(n + dn2/2, u, rho, dt, dx, A) * dt
    dn4 = a_diff(n + dn3, u, rho, dt, dx, A) * dt
    dn = 1/6 * (dn1 + 2*dn2 + 2*dn3 + dn4)
    return n + dn

def a_Eul(n, u, rho, dt, dx, A, f_r):
    kCO = 1.3*10**11 # dm3/mol*s
    kC = 1.3*10**11
    dn = np.zeros((len(n[:,1]), len(n[1,:])))
    R = np.zeros((1,len(u[0,:])))
    RCOx = np.zeros((1,len(u[0,:])))
    RCO = np.zeros((1,len(u[0,:])))
    RC = np.zeros((1,len(u[0,:])))
    s = np.array([[-1], [0], [7/2], [7/2], [5/2], [0], [3/2]])
    # TODO sklad produktow posrednich wybuchu - zmiana objetosci TNT do fazy gazowej
    R[0,:] = A * rho * u * dt / 0.227 * (n[0,:] > 0)
    RCOx[0,:] = n[2,:] * n[5,:]**(1/2) * kCO * dt 
    RC[0,:] = n[3,:] * n[5,:]**(1/2) * kC * dt
    R = np.minimum(R, n[0,:])
    RCO = np.minimum(RCOx, RCOx * n[5,:] / (RCOx + RC))
    RCO[np.isnan(RCO)] = 0
    RC = np.minimum(RC, (RC/2) * n[5,:] / (RCOx + RC))
    RC[np.isnan(RC)] = 0
    # R[0,:] = k * n[0,:] * dt
    # R[0,:] = np.minimum(n[0,:], R)
    # R[0,:] = np.minimum(2/21 * n[3,:], R)
    # R[0,:] = np.minimum(n[0,:], n[3,:]*2/7) 
    n = n + np.matmul(s,R)    
    n[1,:] = n[1,:] + RCO
    n[2,:] = n[2,:] - RCO + RC
    n[3,:] = n[3,:] - RC
    n[5,:] = n[5,:] - 1/2 * RCO - 1/2 * RC
    
    dn[:,0] = \
        - u[:,0] * n[:,0] * dt / dx \
        + np.array([0,0,0,0,0,n_o2,n_n2]) * dt / dx
        # - u[0,0] * (n[:,0] - n[:,1]) * dt / dx \
        # + (-n[:,0] + np.array([0,0,0,0,0,n_o2,n_n2])) * dt / dx
    
    dn[:,1:len(n[0,:])] = \
        - u[:,1:len(n[0,:])] * n[:,1:len(n[0,:])] * dt / dx \
        + u[:,0:len(n[0,:])-1] * n[:,0:len(n[0,:])-1] * dt / dx
        # - u[0,1:len(u[0,:])-1] * (n[:,1:len(n[0,:])-1] - n[:,2:len(n[0,:])]) * dt / dx \
        # + u[0,0:len(u[0,:])-2] * (n[:,0:len(n[0,:])-2] - n[:,1:len(n[0,:])-1]) * dt / dx
    return n + dn

path = "E:/Simulations/Shockwave/"
name = "workspace_no_o2_in_tnt_conservative_advection/"
    ##TODO zamienic obliczanie p na pVm^g = const (wiec 101325 i Vm wynikajace ze stanu podstawowego powietrza!)
Time = 10 #s
Lx = 10
dx = 0.001
dt = 10**(-8)
_rho_air = 1.293 # kg/m3 
_p_ambient = 101325 # Pa
_u_D = 6900 # m/s
r0 = 0.04 # m
_V_init = 4/3 * math.pi * (r0)**3 # m3
m = 183.86 * 10**-3 # kg
_rho0 = m / _V_init # kg/m3
n_tnt = _rho0 * dx**3 / 0.227# mol/mm3
n_o2 = _rho_air * dx**3 * 0.21 / 0.032
n_n2 = _rho_air * dx**3 * 0.79 / 0.028
gamma = 1.4
pvm = 101325 * (dx**3 / (n_o2 + n_n2))**gamma

xx = range(int(Lx/dx)+1)
x = np.zeros((1,len(xx)))
x[0,:] = np.array([a for a in xx]) * dx

u = np.ones((1,len(x[0,:])))
u[0,x[0,:] <= r0] = _u_D # m/s

n = np.zeros((7, len(x[0,:]))) # kolumny TNT (s), CO2, CO, C (s), H2O, O2, N2
n[5,:] = n_o2
n[6,:] = n_n2

rho = np.ones((1,len(x[0,:]))) * _rho_air # kg/m3
# rho[0,x[0,:] <= r0] = _rho0
n[0,x[0,:] <= r0] = n_tnt #* np.array([[0],[7],[5/2],[0],[3/2]])
rho = rho_diff(n, rho, dx)
Vmt = np.zeros((1,len(x[0,:])))
Vmt[0,:] = Vm_calc(n, dx)

p = np.ones((1,len(x[0,:]))) * _p_ambient # Pa
p[0,:] = p_diff(pvm, Vmt, gamma)

mu = 17.06*10**(-6) / 1.293
step = 0
max_step = int(Time*(1/dt))
f_r = 0

df = pd.DataFrame({
        "dx" : dx,
        "dt" : dt,
        "rho air" : _rho_air,
        "p ambient" : _p_ambient,
        "u det" : _u_D,
        "explosive radius" : r0,
        "explosive mass" : m,
        "gamma" : gamma,
        "dynamic viscosity" : mu,
        "max time" : Time
    }, index=[0])
try:
    os.makedirs(path + name, exist_ok=True) 
except OSError as error:
    print(error)
finally:
    df.to_csv(path + name + '/params.csv')

while step < max_step:
    if step % 1000 == 0:
        plot(x, n, p, u, rho, Vmt, step, dt, path+name)
        save_data(x, n, u, p, rho, Vmt, path + name, step)
        
    if step > 0:
        a = 0
        # u[0,0] = 1
        # n[0,0] = 0
        # n[1,0] = 0
        # n[2,0] = 0
        # n[3,0] = n_o2
        # n[4,0] = n_n2
        # p[0,0] = _p_ambient
        # rho[0,0] = _rho_air
    # f_r = f_r + u[0,int(f_r)] * dt    
    n = a_Eul(n, u, rho, dt, dx, dx**2, f_r)
    rho = rho_diff(n, rho, dx)
    Vmt[0,:] = Vm_calc(n,dx)
    p[0,:] = p_diff(pvm, Vmt, gamma)
    u = u_Eul(u, dt, dx, p, rho, f_r)

    step = step + 1