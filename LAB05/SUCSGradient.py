# Optimization for Engineers - Dr.Johannes Hild
# scaled unit central simplex gradient

# Purpose: Approximates gradient of f on a scaled unit central simplex

# Input Definition:
# f: objective class with methods .objective()
# x: column vector in R ** n(domain point)
# h: simplex edge length
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# grad_f_h: simplex gradient
# stenFail: 0 by default, but 1 if stencil failure shows up

# Required files:
# < none >

# Test cases:
# myObjective = multidimensionalObjective()
# x = np.array([[1.02614],[0],[0],[0],[0],[0],[0],[0]], dtype=float)
# h = 1.0e-6
# myGradient = SUCSGradient(myObjective, x, h)
# should return
# myGradient close to [[0],[0],[0],[0],[0],[0],[0],[0]]


import numpy as np
from multidimensionalObjective import multidimensionalObjective

def matrnr():
    # set your matriculation number here
    matrnr = 23533206
    return matrnr


def SUCSGradient(f, x: np.array, h: float, verbose=0):

    if verbose: # print information
        print('Start SUCSGradient...') # print start

    grad_f_h = x.copy() # initialize simplex gradient of f

    # INCOMPLETE CODE STARTS
    
    if hasattr(f, 'objective'):      # test whether f has attribute “objective”
        f_eval = f.objective         # redirect evaluations to f.objective(...)
    else:                            # otherwise f is already a plain callable
        f_eval = f                   # so just use f itself

    n = x.shape[0]                   # extract vector dimension n = len(x)
    grad_f_h[:] = 0.0                # reset gradient storage to all zeros
    for j in range(n):               # loop over each coordinate direction j
        e_j = np.zeros_like(x)       # build zero vector of same shape as x
        e_j[j] = 1.0                 # set its j‑th entry to one → unit vector
        f_plus  = f_eval(x + h * e_j)     # evaluate f at x + h·e_j
        f_minus = f_eval(x - h * e_j)     # evaluate f at x – h·e_j
        grad_f_h[j] = (f_plus - f_minus) / (2.0 * h)  # central diff quotient
    
    # INCOMPLETE CODE ENDS    
    if verbose: # print information
        print('SUCSGradient terminated with gradient =', grad_f_h) # print termination

    return grad_f_h


def SUCSStencilFailure(f, x: np.array, h: float, verbose=0):

    if verbose: # print information
        print('Check for SUCSStencilFailure...') # print start of check

    stenFail = 1 # initialize stencil failure with true

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    
    if hasattr(f, 'objective'):      # check for attribute “objective”
        f_eval = f.objective         # call via that method
    else:                            # otherwise
        f_eval = f                   # keep f as the callable

    fx = f_eval(x)                   # reference value f(x) at current point
    n = x.shape[0]                   # dimension of the decision vector
    for j in range(n):               # test each coordinate direction j
        e_j = np.zeros_like(x)       # construct j‑th unit vector
        e_j[j] = 1.0                 # set its j‑th component to 1
        # If either neighbour’s value is lower, stencil failure did not occur.
        if f_eval(x + h * e_j) < fx or f_eval(x - h * e_j) < fx:
            stenFail = 0             # flag: NO stencil failure
            break                    # exit early once one descent direction found
        
    # INCOMPLETE CODE ENDS
    
    if verbose: # print information
        print('SUCSStencilFailure check returns ', stenFail) # print termination

    return stenFail

# myObjective = multidimensionalObjective()
# x = np.array([[1.02614],[0],[0],[0],[0],[0],[0],[0]], dtype=float)
# h = 1.0e-6
# myGradient = SUCSGradient(myObjective, x, h)
# print(myGradient)