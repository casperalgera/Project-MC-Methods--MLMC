import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.optimize import brentq

sigmasq = 1 #>= 1
lamb = 1   #<= Diam(D) = 1
cov = lambda x,y: sigmasq * np.exp(-abs(x-y)/lamb) 


#We find the nth solution of omega using the periodicity of tan combined with the fact that 1/x**2 -> 0
#Note that the function has been rewritten into a form more suitable for numerical root-finding
#TODO: Turn into precomputed array to speed up calculations.
f = lambda x: (lamb**2*x**2 - 1)*np.sin(x) - 2*lamb*x*np.cos(x)
omega_0 = brentq(f, 0.01, np.pi - np.pi/4)
omega = lambda n: omega_0 if n == 0 else brentq(f, n*np.pi - np.pi/4, n*np.pi + np.pi/2.5)

theta_1D = lambda n: (2*lamb) / (lamb**2 * omega(n)**2 + 1)

def b_1D(n, x):
    normalisation = integrate.quad(lambda t: (np.sin(omega(n)*t) + lamb*omega(n)*np.cos(omega(n)*t))**2, 0, 1, limit=50)[0]
    return (np.sin(omega(n)*x) + lamb*omega(n)*np.cos(omega(n)*x)) / np.sqrt(normalisation)

#We assume a Gaussian random field Z..
def Truncated_KL_Expansion(x, theta, b):
    theta_sum = 0
    m_KL = 0
    while theta_sum < 0.99:
        theta_sum += theta(m_KL)
        print(theta_sum)
        m_KL += 1
    print(m_KL)
    return np.sum([np.sqrt(theta(i))*np.random.normal(0,1)*b(i, x) for i in range(m_KL)])

plt.hist([Truncated_KL_Expansion(-5, theta_1D, b_1D) for _ in range(10)], alpha=0.5)
plt.hist([Truncated_KL_Expansion(5, theta_1D, b_1D) for _ in range(10)], alpha=0.5)
plt.show()


#Sampel from random input filed k(x, \omega)

#Determine number of sample for each grid size.

#Solve Q + k nabla p = G
# nabla q = 0 in D \susbet \R^d