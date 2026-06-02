import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import scipy.integrate as integrate
import scipy
import error_analysis
from tqdm import tqdm

sigmasq = 1 #>= 1
lamb = 1   #<= Diam(D) = 1
precompute_threshold = 49 #How many eigenfunctions and -values of the KL-expansion we precompute.
klcutoff = 0.99 #The fraction of the variance we require to be captured by the KL-expansion. Higher fraction warrants a higher precomput threshold. 
alpha=2#Order of convergence of the Finite Volume Method
error_proportionality_constant=1.#E(Q_L-Q) \approx error_proportionality_constant * M^-alpha
eps=0.00001#Desired precision level
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


#Setup function for generating the coordinates of the grid points
gen_grid = lambda N: np.linspace(0.5/N, 1.-0.5/N, N, True)

def Truncated_KL_Expansion(M_vals, theta, b):
    '''
    Sample from a Gaussian random field Z
    using the truncated Karhunen-Loève expansion on some grids of sizes M
    Parameters:
    --------------------
    M_vals: [int]
        The sizes of the grids on which k needs to be sampled
    theta: [function]
        Function returning the nth eigenvalue of the covariance operator
         with kernel C(x,y) =  sigma^2 exp(-||x-y||^2/lambda).
    b: [function]
        Function returning the nth normalised eigenfunctions of the 
        covariance operator. 
    '''
    if isinstance(M_vals, int):
        M_vals_arr=[M_vals]
    else:
        M_vals_arr=M_vals
    theta_sum = 0
    m_KL = 0
    while theta_sum < klcutoff*sigmasq:
        theta_sum += theta(m_KL)
        m_KL += 1
    xi_n = np.random.normal(0,1, size= m_KL)
    results=[]
    for M in M_vals_arr:
        x_vals=np.linspace(0.5/M, 1.-0.5/M, M, True)
        results.append(np.sum([np.sqrt(theta(i))*xi_n[i]*b(i, x_vals) for i in range(m_KL)], axis=0))
    return results

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
    RHS=f*(h**2)
    RHS[0]+=2.*k[0]
    p=scipy.sparse.linalg.spsolve(scipy.sparse.csc_array(A), RHS)
    return p

#Convergence analysis for the Finite Volume Method
def approximate_solution(m):
    '''
    Find the approximate solution to -d/dx(k dp/dx)=f
    for k=1+x, p=x-x^2, f=1+4x using the finite volume method
    '''
    x_vals=np.linspace(0.5/m, 1.-0.5/m, m, True)
    k=1.+x_vals
    f=-4.*x_vals
    return FVM(f, k)


def exact_solution(m):
    '''
    Find the exact solution to -d/dx(k dp/dx)=f
    for k=1+x, p=x-x^2, f=1+4x
    '''
    x_vals=np.linspace(0.5/m, 1.-0.5/m, m, True)
    return np.square(1-x_vals)

#m_vals=16*np.exp2(np.arange(12)).astype(int)
#error_analysis.compare_to_exact_solution(approximate_solution, exact_solution, m_vals)


# plt.hist([Truncated_KL_Expansion(-5, theta_1D, b_1D) for _ in range(10000)], alpha=0.5)
# plt.hist([Truncated_KL_Expansion(5, theta_1D, b_1D) for _ in range(10000)], alpha=0.5)
# plt.show()

def Compute_Q(grid, grid_size):
    k_grid = np.array([k(x) for x in grid])
    p = FVM(f(grid), k_grid)
    return 2*grid_size*k_grid[-1]*p[-1]


f = lambda x: x*np.exp(-(x**2)/0.25) 

def k(M_vals):
    '''
    Sample from the random field k and return its value on grids with grid sizes M_vals
    Parameters:
    --------------
    M_vals: [int] or int
        Grid sizes
    Returns:
    0, 1, ..., len(M_vals): The sample of k evaluated at grids with grid sizes M_vals[0], M_vals[1], ..., M_vals[len(M_vals)-1]
    '''
    if isinstance(M_vals, int):
        M_vals_arr=[M_vals]
    else:
        M_vals_arr=M_vals
    Z=Truncated_KL_Expansion(M_vals_arr, theta_1D, b_1D)
    result=[]
    for i in range(len(M_vals_arr)):#Cant use numpy instead of this loop, since the arrays are not the same shape
        result.append(np.exp(Z[i]))
    return result

def draw_Y_L_samples(M0, s, sample_num, L):
    samples=np.empty(sample_num)
    M_curr = int(M0*(s**L))
    M_prev = int(M0*(s**(L-1)))
    x_curr = gen_grid(M_curr)
    x_prev = gen_grid(M_prev)
    for i in tqdm(range(sample_num), desc="Generating Y_L samples with new grid size " + str(M_curr) ,leave=False):
        (k_grid_curr, k_grid_prev) = k([M_curr, M_prev])
        p_curr = FVM(f(x_curr), k_grid_curr)
        p_prev = FVM(f(x_prev), k_grid_prev)
        samples[i] = 2*M_curr*k_grid_curr[-1]*p_curr[-1] - 2*M_prev*k_grid_prev[-1]*p_prev[-1]
        if np.abs(samples[i])>0.002:
            print("problem")
    return samples

def draw_Q_L_samples(M, sample_num):
    '''
    Draw sample_num samples on a grid of size M
    '''
    samples=np.empty(sample_num)
    x_vals= gen_grid(M)
    for i in tqdm(range(sample_num), desc="Generating Y_0=Q_0 samples",leave=False):
        k_grid = k(M)[0]
        p = FVM(f(x_vals), k_grid)
        samples[i] = 2*M*k_grid[-1]*p[-1]
    return samples
def MLMC(Nmin, M0, s):
    '''
    Nmin samples at L
    M0 initial grid size
    s grid scaling
    '''
    converged = False
    #First Monte Carlo level is different
    L = 0
    M = M0*(s**L)#Number of grid points
    Y = [draw_Q_L_samples(M, Nmin)]#Setup sample array
    
    VY_L = [np.var(Y[0])]#Estimate variance
    #Setup array for proportionality constants for N
    N_proportion=[np.sqrt(VY_L[0]/(s**L))]
    while not converged:
        L = L+1
        M = M0*(s**L)
        #Add new array for storing samples and compute samples for the new MLMC level
        Y.append(draw_Y_L_samples(M0, s, Nmin, L))        
        #Estimate variance of new level
        VY_L.append(np.var(Y[L]))
        current_prop_const=np.sqrt(VY_L[L]/(s**L))
        N_proportion.append(current_prop_const)#Add new propoportionality constant
        #Compute how many samples are now needed
        N_vals=np.ceil(np.array(N_proportion)/current_prop_const*Nmin).astype(int)
        #Add extra samples for other levels
        if Y[0].size<N_vals[0]:
            Y[0]=draw_Q_L_samples(M0, N_vals[0]-Y[0].size)
        for update_L in range(1, L):
            if Y[update_L].size<N_vals[update_L]:
                Y[update_L]=np.concatenate((Y[update_L], draw_Y_L_samples(M0, s, N_vals[update_L]-Y[update_L].size, update_L)))
        #Test for convergence
        var=np.sum(np.array(VY_L)/N_vals)#Estimator of variance of Q_MLMC 
        print("Variance is: " + str(var))
        print("err is " + str(error_proportionality_constant*M**(-alpha)))
        if var+error_proportionality_constant*M**(-alpha)<eps:
            converged = True
    #Return MLMC estimate
    result=0.
    for l in range(L+1):
         result+=np.average(Y[l])
    return (result, Y, N_vals)


draw_Y_L_samples(16, 2, 1000, 2)
#MLMC(30, 16, 2)
'''
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
plt.show()'''