import math
import numpy as np

from params import Params, pressure
from Advection import Advection_Simple
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

''' OLD APPROACH / ANOTHER APPROACH
    stworzyc strukture zachowujaca parametry czola fali
        jako parametry inicjujace przyjac wartosci w x = 0
        Us = Udet
        
    jako element poczatkowy zmiana parametrow downstream fali (x+1)
        przyjac jako skok parametry fali
    
    kolejno zmiana parametrow w upstream fali (x) korzystajac z adwekcji i NS
    
    na koniec nalozyc poprawke na parametry czola fali korzystajac z zauktualizowanych wartosci upstream i downstream    
'''

''' NEW APPROACH - zle wzory!
    jump condition wyznaczamy z całkowych postaci równań zahcowania
        [.] jump value (.1 - .0)
    1. conservation of mass - predkosc w downstream - przed fala
        dx/dt * [rho] = [rho*u]
    2. conservation of momentum - cisnienie w downstream
        dx/dt * [rho * u] = [rho * u^2 + p]
    3. conservation of energy - temperatura
        dx/dt * [rho * H] = [rho * u * H] - H - entalpia? jaka?
'''

def jump(r, u, p, rho, dt, dx, idx, H=0, Cp=0):
    # idx = int(math.floor(r*(int(1/Params.dx))))
    # print(idx)
    # us = (rho[idx]*u[idx] - rho[idx+1]*u[idx+1])/(rho[idx]-rho[idx+1]) # mass conservation
    us = (u[idx]*rho[idx]-u[idx+1]*rho[idx+1])/(rho[idx]-rho[idx+1])
    # ps = p[idx] + rho[idx] * u[idx]**2 - ( rho[idx+1] * u0**2 + ( rho[idx] * u[idx]**2 - rho[idx+1] * u0**2 ) * dx/dt )
    ps = p[idx] - rho[idx+1] * (us - u[idx+1])*(u[idx] - u[idx+1])
    dT = 0
    # dT = Cp / (H[idx] - ( rho[idx] * H[idx] * (u[idx]-dx/dt) ) / ( rho[idx+1] * (u0 - dx/dt) ))
    return [us, ps, dT]


