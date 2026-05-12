import numpy as np
import scipy
import matplotlib.pyplot as plt

#This code is meant for verifying FDM schemes

#It was written earlier for a different subject, so not all functions will be used   



def power(x, A, p):
    ''' 
    Exponential function f(x)=A*x^p 
    Parameters:
    -----------
    x: float
        independent variable
    A: float
    p: float
    '''
    return A*np.power(x, p)

def linear(x, a, p):
    '''
    Linear function f(x)=a+px
    Parameters
    -------------
    x: float
        indepent variable
    a: float
        Starting value
    p: float
        slope
    '''
    return a+p*x

#Norms for computing errors:
def l_inf(approx_solution, exact_solution):
    '''
    Compute the L_inf norm of the difference of the arrays approx_sol and exact_sol
    Parameters:
    approx_sol
    '''
    return np.max(np.abs(approx_solution-exact_solution))


def L_2_norm_approx(approx_sol, exact_sol, M):
    '''
    Compute the L_2 error of the approximate solution approximately using the mass matrix M
    (Apparently) ||approx_sol-exact_sol||_L_2  \approx (approx_sol-exact_sol)^T M (approx_sol-exact_sol)
    Parameters:
    --------------------------------------
    approx_sol: [float]
        - The array U for the approximate solution
    exact_sol: [float]
        - The exact solution at the nodes
    M: [[float]]
        - The mass matrix
    '''
    err=approx_sol-exact_sol
    return np.transpose(err)@M@err
    

def compare_to_exact_solution(approximate_solutions, exact_solution, N_vals, x_label="h", y_label=r"$||u_{exact}-u_{numerical}||_{\infty}$", box_length=1, args=[], extra_args=[], extra_args_exact=[], norm=l_inf):
    '''
    Make a log-log plot of the L^1/L^2 error of the numerical solution
    
    Parameters
    -------------
    approximate_solutions: function or [function]
        The functions that computes the approximate solution, the first parameter should be N, the number of grid points. It is also possible to compare multiple methods by giving an array of functions
    exact_solution: function
        The function that computes the exact solution
    N_vals: array of ints
        The numbers of grid points the error should be computed for
    x_label: string
        The label for the x axis
    y_label : string
        The label for the y axis
    box_length: float
        the size of the box
    args: []
        Extra arguments for both approximate_solution and exaxt_solution
    extra_args: []
        Extra arguments for only approximate_solution, if multiple methods are to be tested, extra_args should be an array with the same size as approximate_function. Then the i-th arguments are passed to the i-th function
    extra_args_exaxt: []
        Extra arguments only for exact_solutions
    norm: function: ([float], [float]) -> [0, +infty)
        A function that computes the norm between the exact and approximate solutions. This does not have to be a norm, it can also be an approximation of a norm. The first argument should be the approximate solution and the second the exact solution. 
        
    '''
    #Put the scheme in an array if only one scheme was given and it was niot in an array already
    if callable(approximate_solutions):
        approximate_solutions=[approximate_solutions]
        extra_args=[extra_args]
    #Initialize error array
    err=np.empty((len(approximate_solutions), N_vals.size))
    # Compute L^1 error of numerical solutions
    for j in range(len(approximate_solutions)):
        for i in range(N_vals.size):
            err[j, i]=norm(approximate_solutions[j](N_vals[i], *args, *extra_args[j]), exact_solution(N_vals[i], *args, *extra_args_exact))
            print([i, err[j, i]])
    #log-log plot the error
    fig=plt.figure()
    ax=fig.add_subplot()
    ax.set_xscale("log") 
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    h_vals=box_length/N_vals
    #Compute the order of convergence
    for j in range(len(approximate_solutions)):
        fit=scipy.optimize.curve_fit(linear, np.log(h_vals), np.log(err[j]))
        ax.plot(h_vals, power(h_vals, np.exp(fit[0][0]), fit[0][1]), label="p="+str(fit[0][1]))
        ax.scatter(h_vals, err[j])
    plt.legend()
    plt.show()
    print(err)
    print(fit)
    
def set_as_array(x, size, is_complex=True, params=[]):
    '''
    Checks if x is an array and turns x into an array if it was not already
    Parameters
    -------------
    x: any
        The value that should be an array. If it is an array, nothing will happen. If x is a function, x(size) will be returned. If it us anything else an array filled with x will be returned
    size: int
        The size of the array
    is_complex: bool
        Should the output be a complex number
    params: []
        Array of parameters to pass to x if x is a function
    '''
    if isinstance(x, (list, np.ndarray)):
        return x
    elif callable(x):
        return x(size, *params)
    elif is_complex:
        return np.repeat(x+0j, size)
    else:
        return np.repeat(x, size)