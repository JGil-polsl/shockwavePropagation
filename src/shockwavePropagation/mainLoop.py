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

def volumeCalculation(k1, k2, k3, TNT, dt, init):
    dC = init[0]
    dH2O = init[1]
    dN2 = init[2]
    dCO = init[3]
    dCO2 = init[4]
    
    dC = (k1*(TNT**7)-k2*dCO)*dt
    dCO = (k2*dC - k3*dCO2)*dt
    dCO2 = (k3*dCO)*dt
    dH2O = (k1*(TNT**(5/2)))*dt
    dN2 = (k1*(TNT**(3/2)))*dt
                                                                                        
def mainLoop():
    print('a')
    