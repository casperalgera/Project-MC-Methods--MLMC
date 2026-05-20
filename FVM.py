import numpy as np
import scipy
import error_analysis
import matplotlib.pyplot as plt
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

#approximate_solution(100)
m_vals=16*np.exp2(np.arange(12)).astype(int)
error_analysis.compare_to_exact_solution(approximate_solution, exact_solution, m_vals)