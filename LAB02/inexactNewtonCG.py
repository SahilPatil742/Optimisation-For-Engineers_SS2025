# Optimization for Engineers - Dr.Johannes Hild
# inexact Newton CG

# Purpose: Find xmin to satisfy norm(gradf(xmin))<=eps
# Iteration: x_k = x_k + t_k * d_k
# d_k starts as a steepest descent step and then CG steps are used to improve the descent direction until negative curvature is detected or a full Newton step is made.
# t_k results from Wolfe-Powell

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# dA = directionalHessApprox(f, x, d) from directionalHessApprox.py
# t = WolfePowellSearch(f, x, d) from WolfePowellSearch.py

# Test cases:
# myObjective = noHessianObjective()
# x0 = np.array([[-0.01], [0.01]])
# xmin = inexactNewtonCG(myObjective, x0, 1.0e-6, 1)
# should return
# xmin close to [[0.26],[-0.21]]

import numpy as np
import WolfePowellSearch as WP
import directionalHessApprox as DHA 
from noHessianObjective import noHessianObjective

def matrnr():
    # set your matriculation number here
    matrnr = 23533206
    return matrnr


def inexactNewtonCG(f, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0: # check for positive eps
        raise TypeError('range of eps is wrong!')

    if verbose: # print information
        print('Start inexactNewtonCG...') # print start

    countIter = 0 # counter for number of loop iterations
    xk = x0.copy() # initialize starting iteration
    
    
    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    
    grad_fk = f.gradient(xk)  # Compute gradient at xk
    norm_grad = np.linalg.norm(grad_fk)  # Compute norm of gradient
    eta_k = np.min([0.5, np.sqrt(norm_grad)]) * norm_grad  # Set CG tolerance eta_k

    while norm_grad > eps:  # Main loop: until gradient norm is below tolerance
        xj = xk.copy()  # Initialize xj to current point xk
        rj = grad_fk.copy()  # Set residual rj to current gradient
        dj = -rj.copy()  # Initialize search direction as steepest descent

        while np.linalg.norm(rj) > eta_k:  # Inner CG loop: until residual small enough
            dA = DHA.directionalHessApprox(f, xk, dj)  # Approximate Hessian-vector product
            rhoj = (dj.T @ dA).item()  # Compute curvature value

            if rhoj <= eps * np.linalg.norm(dj) ** 2:  # Check curvature condition
                break  # Break if curvature condition fails (non-positive)

            tj = np.linalg.norm(rj) ** 2 / rhoj  # Compute CG step size tj
            xj = xj + tj * dj  # Update xj with CG step
            r_old = rj.copy()  # Store old residual
            rj = r_old + tj * dA  # Update new residual using dA

            beta_j = np.linalg.norm(rj) ** 2 / np.linalg.norm(r_old) ** 2  # Compute CG beta
            dj = -rj + beta_j * dj  # Update direction using CG formula

        if not np.array_equiv(xj, xk):  # If CG made progress
            dk = xj - xk  # Set Newton direction as CG step
        else:
            dk = -grad_fk.copy()  # Fall back to steepest descent if no progress

        tk = WP.WolfePowellSearch(f, xk, dk)  # Perform line search along direction dk

        xk = xk + tk * dk  # Update xk with step
        grad_fk = f.gradient(xk)  # Recompute gradient at new point
        norm_grad = np.linalg.norm(grad_fk)  # Update norm of gradient
        eta_k = np.min([0.5, np.sqrt(norm_grad)]) * norm_grad  # Recompute CG tolerance

        countIter += 1  # Increment iteration count

    
    # INCOMPLETE CODE ENDS
    
    if verbose: # print information
        stationarity = np.linalg.norm(f.gradient(xk)) # store stationarity value
        print('inexactNewtonCG terminated after ', countIter, ' steps with norm of gradient =', stationarity) # print termination with stationarity value

    return xk
