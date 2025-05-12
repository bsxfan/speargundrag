"""
This is a simple model for how a speargun spear decelerates through the 
water after leaving the gun at time t=0, when the initial ('muzzle') 
velocity is v0. The model parameters are just v0 and k, a deceleration
coefficient that is dependent on the spear dimensions and mass.

The only assumption we make is that drag is proportional to speed. This
gives a differential equation, which when solved shows that the speed 
decays exponentially as a function of time and linearly as a function of
distance.

The model is equipped with the ability to estimate its parameters (v0, k)
from given, noisy measurement data.

By running this module as a script, you can generate an example plot that
compares naive and model-based speed estimates.
"""
import numpy as np
from numpy import ndarray
from scipy.optimize import minimize


class Model:
    def __init__(self, v0:float, k:float):
        """
        model parameters:
            
            v0: initial speed at time t=0 [m/s]
        
            k: drag coefficient (always positive) [1/s]
        
        Both parameters can be estimated using Model.fit() 
        """
        self.v0 = v0
        self.k = k
        self.x_max = v0/k
        
    def x_of_t(self, t: float | ndarray):
        """
        Computes distance travelled [m] at time t.
        """
        return -self.x_max*np.expm1(-self.k*t)
    
    def v_of_t(self, t: float | ndarray):
        """
        Computes speed [m/s] at time t.
        """
        return self.v0*np.exp(-self.k*t)

    def v_of_x(self, x: float | ndarray):
        """
        Computes speed [m/s] at distance x < x_max. 
        
        (At infinite time, we get x = x_max = v0/k.)
        """
        assert (x < self.x_max).all(), f'We need x < x_max = {self.x_max}'
        return self.v0 - self.k*x

    @classmethod
    def fit(cls, x:ndarray, t:ndarray) -> 'Model':
        """
        Provide measuremsnts that includes distance travelled and time elapsed.
        
        If you estimate from video, choose a video frame where the spear has 
        already left the gun. Define the time and spear position of this frame
        as t=0 and x=0. The measurements provided here are relative to that
        origin. 
        
        The estimated model parameter v0 is the initial velocity at x=0
        and t=0.
        
        """
        def obj(params):
            model = Model(*np.exp(params))
            return ((x-model.x_of_t(t))**2).mean()
        params0 = np.zeros(2)
        res = minimize(obj, params0, method='Nelder-Mead')    
        assert res.success, 'fit failed'
        print('model fit RMS error:', res.fun)
        v0, k = np.exp(res.x)
        return cls(v0, k)
    
    
    
class Spear:
    def __init__(self, drag_coefficient:float = 89.5, 
                       density:float = 7.75, 
                       diameter:float =7e-3, 
                       length:float = 1.7):
        self.drag_coefficient = drag_coefficient
        self.density = density
        self.diameter = diameter
        self.length = length
        self.x_area = x_area = np.pi * diameter**2
        self.volume = vol =  x_area * length
        self.mass = mass = density * 1_000 * vol
        self.area = area = 2 * np.pi * diameter * length # exclude x_area
        self.k = drag_coefficient * area / mass
        
        
    def launch(self, v0) -> Model:    
        return Model(v0, self.k)
        
        
    @classmethod
    def drag_coefficient_given_k(cls, k:float=3.3, spear:'Spear'=None) -> float:
        """
        k = c * area / mass
        """
        if spear is None: spear = Spear()
        return k*spear.mass / spear.area
        
class Target:
    def __init__(self, penetration_pressure:float = 7e6):
        self.penetration_pressure = penetration_pressure
        
    def penetration_depth(self, speed, spear:Spear = Spear()) -> float:
        energy = 0.5 * spear.mass * speed **2
        force = spear.x_area * self.penetration_pressure
        depth = energy / force
        return depth
    
    def min_required_speed(self, spear:Spear = Spear(), 
                                 penetration_depth: float = 0.1) -> float:
        force = spear.x_area * self.penetration_pressure
        energy = force * penetration_depth
        speed = np.sqrt(2*energy/spear.mass)
        return speed
    
    
    @classmethod
    def penetration_pressure_estimate(cls, speed:float = 10.0, 
                                           penetration_depth:float = 0.1, 
                                           spear:Spear = Spear()) -> float:
        energy = 0.5 * spear.mass * speed **2
        force = energy / penetration_depth
        pressure = force / spear.x_area
        return pressure
        
        
        
        
        
        
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    v0, k = 25.0, 3.3
    model0 = Model(v0, k)
    
    
    tmax = 1.0
    t = np.linspace(0, tmax, 200)
    v = model0.v_of_t(t)
    x = model0.x_of_t(t)
    
    plt.plot(t, v, label='speed [m/s]')
    plt.plot(t, x, label='distance [m]')
    plt.legend()
    plt.grid()
    plt.xlabel('time [s]')
    plt.show()
    
    nsamples = 10
    tss = np.linspace(0, tmax, nsamples)
    
    plt.figure()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    noise_rms = 1/20
    for i in range(1,100):
        noise = np.random.randn(nsamples) * noise_rms
        xn = model0.x_of_t(tss) + noise
        
        model1 = Model.fit(xn, tss)
        v1 = model1.v_of_t(t)
        
        delta_t = tss[1:] - tss[:-1]
        delta_x = xn[1:] - xn[:-1]
        vn = delta_x/delta_t
        tn = (tss[1:] + tss[:-1])/2
        
        
        plt.plot(t, v1, '.', color=colors[1])
        plt.plot(tn, vn, '.g')
        plt.plot(tss, xn, '.r')


    plt.plot(t, v1, '.', color=colors[1], label='model speed estimate [m/s]')
    plt.plot(tn, vn, '.g', label='naive speed estimate [m/s]')
    plt.plot(tss, xn, '.r', label='measured distance [m]')

    plt.plot(t, v, 'k', label='actual speed [m/s]')
    plt.plot(t, x, 'b', label='actual distance [m]')
    plt.legend()
    plt.grid()
    plt.xlabel('time [s]')
    plt.title(f'distance measurement error rms: {noise_rms*100} cm')
    plt.show()
    
    
    

    
            
            
            
            
        
