'''

Thermal model wave propagation
    Based on NS equation end Euler model (or RK4)
    
    parameters:
        pressure - clapeyron Equation
        density - molar distribution per volume
        velocity - NS equation
        molar volume - molar distribution
        volume        
        temperature - reaction hear
        molar distribution - Advection equation
        
'''

class molarMass():
    TNT = 7*12.011 + 5*1.008 + 3*14.007 + 6*15.999
    C = 12.011
    CO = 12.011 + 15.999
    CO2 = 12.011 + 2*15.999
    H2O = 2*1.008 + 15.999
    N2 = 2*14.007

class Params():
    dx = 10**(-3) # m
    dt = 10**(-9) # s
    
    VOLUME = dx**3 # m3
    TEMPERATURE = 293 # K
    PRESSURE = 101350 # Pa
    DENSITY = 0 # kg/m3
    
def init():
    return    
    
def main():
    return
    
if __name__ == "__main__":
    main()