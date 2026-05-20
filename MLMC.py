import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.optimize import brentq

sigmasq = 1 #>= 1
lamb = 1   #<= Diam(D) = 1


precompute_threshold = 49 #How many eigenfunctions and -values of the KL-expansion we precompute.
klcutoff = 0.99 #The fraction of the variance we require to be captured by the KL-expansion. Higher fraction warrants a higher precomput threshold. 


'''
We find the nth solution of the equation tan(omega) = (2lambdaomega)/(lambda^2omega^2 - 1) using the periodicity of tan.
#The equation has been rewritten into a form more suitable for numerical root-finding
#We precompute omega up to the precompute threshold and approximate for larger values of n using the fact that 1/x**2 -> 0.
#Using this omega we define a function theta_1D for the eigenvalues in the KL-expansion.
'''
f = lambda x: (lamb**2*x**2 - 1)*np.sin(x) - 2*lamb*x*np.cos(x)
om = np.concatenate([[brentq(f, 0.01, np.pi - np.pi/4)], [brentq(f, n*np.pi - np.pi/4, n*np.pi + np.pi/2.5) for n in range(1, precompute_threshold + 1)]])
omega = lambda n: om[n] if n <= precompute_threshold else n*np.pi
theta_1D = lambda n: (2*lamb) / (lamb**2 * omega(n)**2 + 1)

'''
Define and normalise the eigenfunctions b_1D. We precompute the normalisation up to the precompute threshold.
'''
normalisation = [integrate.quad(lambda t: (np.sin(omega(n)*t) + lamb*omega(n)*np.cos(omega(n)*t))**2, 0, 1, limit=50)[0] for n in range(precompute_threshold + 1)]
norm = lambda n: np.sqrt(normalisation[n]) if n <= precompute_threshold else np.sqrt(integrate.quad(lambda t: (np.sin(omega(n)*t) + lamb*omega(n)*np.cos(omega(n)*t))**2, 0, 1, limit=50)[0])
b_1D = lambda n, x: (np.sin(omega(n)*x) + lamb*omega(n)*np.cos(omega(n)*x)) / norm(n)

def Truncated_KL_Expansion(x, theta, b):
    '''
    Sample from a Gaussian random field Z
    using the truncated Karhunen-Loève expansion 
    Parameters:
    --------------------
    x: [float]
        Point in space from which to sample the random field.
    theta: [function]
        Function returning the nth eigenvalue of the covariance operator
         with kernel C(x,y) =  sigma^2 exp(-||x-y||^2/lambda).
    b: [function]
        Function returning the nth normalised eigenfunctions of the 
        covariance operator. 
    '''
    theta_sum = 0
    m_KL = 0
    while theta_sum < klcutoff*sigmasq:
        theta_sum += theta(m_KL)
        m_KL += 1
    return np.sum([np.sqrt(theta(i))*np.random.normal(0,1)*b(i, x) for i in range(m_KL)])

plt.hist([Truncated_KL_Expansion(-5, theta_1D, b_1D) for _ in range(10000)], alpha=0.5)
plt.hist([Truncated_KL_Expansion(5, theta_1D, b_1D) for _ in range(10000)], alpha=0.5)
plt.show()


#Sampel from random input filed k(x, \omega)

#Determine number of sample for each grid size.

#Solve Q + k nabla p = G
# nabla q = 0 in D \susbet \R^d

def FVM(f, k):
    '''
    Solve the equation:
    nabla * (k nabla p)=f
    for p, where f = nabla g for some g, using the Finite Volume Method
    All arrays are of some size J and correspond to the centers of the cells
    Parameters:
    --------------------
    f: [float]
        Source term
    k: [float]
        Hydraulic conductivity
    '''