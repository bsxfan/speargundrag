import numpy as np
from scipy.optimize import minimize


class Model:
    def __init__(self, v0, k):
        self.v0 = v0
        self.k = k
        self.x_max = v0/k
        
    def x_of_t(self, t):
        return -self.x_max*np.expm1(-self.k*t)
    
    def v_of_t(self, t):
        return self.v0*np.exp(-self.k*t)

    def v_of_x(self, x):
        assert (x < self.x_max).all(), f'We need x < x_max = {self.x_max}'
        return self.v0 - self.k*x

    @classmethod
    def fit(cls, x, t):
        def obj(params):
            model = Model(*np.exp(params))
            return ((x-model.x_of_t(t))**2).mean()
        params0 = np.zeros(2)
        res = minimize(obj, params0, method='Nelder-Mead')    
        assert res.success, 'fit failed'
        print('model fit RMS error:', res.fun)
        v0, k = np.exp(res.x)
        return cls(v0, k)
        
    
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
    noise = np.random.randn(nsamples)/20
    xn = model0.x_of_t(tss) + noise
    
    model1 = Model.fit(xn, tss)
    v1 = model1.v_of_t(t)
    
    delta_t = tss[1:] - tss[:-1]
    delta_x = xn[1:] - xn[:-1]
    vn = delta_x/delta_t
    tn = (tss[1:] + tss[:-1])/2
    
    
    plt.plot(t, v, 'k', label='actual speed [m/s]')
    plt.plot(t, x, label='actual distance [m]')
    plt.plot(t, v1, '--', label='model speed estimate [m/s]')
    plt.plot(tn, vn, '.', label='naive speed estimate [m/s]')
    plt.plot(tss, xn, '.', label='measured distance [m]')
    plt.legend()
    plt.grid()
    plt.xlabel('time [s]')
    plt.show()
    
    
    

    
            
            
            
            
        
