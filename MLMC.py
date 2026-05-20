import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import scipy.integrate as integrate
import scipy
import error_analysis

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
g = lambda x: (lamb**2*x**2 - 1)*np.sin(x) - 2*lamb*x*np.cos(x)
om = np.concatenate([[brentq(g, 0.01, np.pi - np.pi/4)], [brentq(g, n*np.pi - np.pi/4, n*np.pi + np.pi/2.5) for n in range(1, precompute_threshold + 1)]])
omega = lambda n: om[n] if n <= precompute_threshold else n*np.pi
theta_1D = lambda n: (2*lamb) / (lamb**2 * omega(n)**2 + 1)

'''
Define and normalise the eigenfunctions b_1D. We precompute the normalisation up to the precompute threshold.
'''
normalisation = [integrate.quad(lambda t: (np.sin(omega(n)*t) + lamb*omega(n)*np.cos(omega(n)*t))**2, 0, 1, limit=50)[0] for n in range(precompute_threshold + 1)]
norm = lambda n: np.sqrt(normalisation[n]) if n <= precompute_threshold else np.sqrt(integrate.quad(lambda t: (np.sin(omega(n)*t) + lamb*omega(n)*np.cos(omega(n)*t))**2, 0, 1, limit=50)[0])
b_1D = lambda n, x: np.array([(np.sin(omega(n)*a) + lamb*omega(n)*np.cos(omega(n)*a)) / norm(n) for a in x])

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
    xi_n = np.random.normal(0,1, size= m_KL)
    return np.sum([[np.sqrt(theta(i))*xi_n[i]*b(i, x)] for i in range(m_KL)], axis=0)


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
    J=k.size
    h=1/J
    #Step 1) Approximate k on grid edges using harmonic averages
    k_shifted=2./(1./k[1:]+1./k[:-1])#0.5*(f[1:]+f[:-1])
    #Step 2)Setup system matrix
    A_diag=np.concatenate([
        [2.*k[0]+k_shifted[0]],
        k_shifted[1:]+k_shifted[:-1],
        [2.*k[-1]+k_shifted[-1]]
    ])
    A=scipy.sparse.diags((-k_shifted, A_diag, -k_shifted), [-1, 0, 1], [J, J])
    p=scipy.sparse.linalg.spsolve(scipy.sparse.csc_array(A), f*(h**2))
    return p


def approximate_solution(m):
    '''
    Find the approximate solution to -d/dx(k dp/dx)=f
    for k=1+x, p=x-x^2, f=1+4x using the finite volume method
    '''
    x_vals=np.linspace(0.5/m, 1.-0.5/m, m, True)
    k=1.+x_vals
    f=1.+4.*x_vals
    return FVM(f, k)


def exact_solution(m):
    '''
    Find the exact solution to -d/dx(k dp/dx)=f
    for k=1+x, p=x-x^2, f=1+4x
    '''
    x_vals=np.linspace(0.5/m, 1.-0.5/m, m, True)
    return x_vals-np.square(x_vals)

# approximate_solution(100)
# plt.hist([Truncated_KL_Expansion(-5, theta_1D, b_1D) for _ in range(10000)], alpha=0.5)
# plt.hist([Truncated_KL_Expansion(5, theta_1D, b_1D) for _ in range(10000)], alpha=0.5)
# plt.show()
# m_vals=16*np.exp2(np.arange(12)).astype(int)
# error_analysis.compare_to_exact_solution(approximate_solution, exact_solution, m_vals)
def Compute_Q(grid, grid_size):
    k_grid = np.array([k(x) for x in grid])
    p = FVM(f(grid), k_grid)
    return 2*grid_size*k_grid[-1]*p[-1]


f = lambda x: x*np.exp(-(x**2)/0.25) 
k = lambda x: np.exp(Truncated_KL_Expansion(x, theta_1D, b_1D))
def MLMC(Nmin, M0, s):
    '''
    Nmin samples at L
    M0 initial grid size
    s grid scaling
    '''
    converged = False
    L = 0
    Y = [np.empty(Nmin)]
    M = M0*(s**L)
    gen_grid = lambda N: np.linspace(0.5/N, 1.-0.5/N, N, True)
    x_vals= gen_grid(M)
    for i in range(Nmin):
        k_grid = np.array([k(x) for x in x_vals])
        p = FVM(f(x_vals), k_grid)
        Y[0][i] = 2*M*k_grid[-1]*p[-1]
    VY_L = np.var(Y[0])
    const = Nmin/np.sqrt(VY_L)
    N = np.array([np.ceil(const * np.sqrt(VY_L))])
    while not converged:
        L = L+1
        Y.append([np.empty(Nmin)])
        M_curr = M0*(s**L)
        M_prev = M0*(s**L)
        x_curr = gen_grid(M_curr)
        x_prev = gen_grid(M_prev)
        for i in range(Nmin):
            k_grid_curr = np.array([k(x) for x in x_curr])
            k_grid_prev = np.array([k(x) for x in x_prev])
            p_curr = FVM(f(x_curr), k_grid_curr)
            p_prev = FVM(f(x_prev), k_grid_prev)
            Y[L][i] = 2*M*(k_grid_curr[-1]*p_curr[-1] - 2*M*k_grid_prev[-1]*p_prev[-1])
        converged = True
#MLMC(10, 2)

x_vals = np.linspace(-5, 5, 100)
n_samples = 1
plt.figure()
for _ in range(n_samples):
    z_vals = Truncated_KL_Expansion(x_vals, theta_1D, b_1D)[0]
    plt.plot(x_vals, z_vals, marker='o', linestyle='-', alpha=0.4)
plt.xlabel('x')
plt.ylabel('Z(x)')
plt.title(f'{n_samples} samples from truncated KL expansion')
plt.grid(True)
plt.show()