# Optimization for Engineers - Dr.Johannes Hild
# Wolfe-Powell line search

# Purpose: Find t to satisfy f(x+t*d)<=f(x) + t*sigma*gradf(x).T@d
# and gradf(x+t*d).T@d >= rho*gradf(x).T@d

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x: column vector in R ** n(domain point)
# d: column vector in R ** n(search direction)
# sigma: value in (0, 1 / 2), marks quality of decrease. Default value: 1.0e-3
# rho: value in (sigma, 1), marks quality of steepness. Default value: 1.0e-2
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# t: t is set, such that t satisfies both Wolfe - Powell conditions

# Required files:
# < none >

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.01], [1]])
# d = np.array([[1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=1

# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.2], [1]])
# d = np.array([[0.1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=16

# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-0.2], [1]])
# d = np.array([[1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=0.25

import numpy as np
from simpleValleyObjective import simpleValleyObjective

def matrnr():
    # set your matriculation number here
    matrnr = 23533206
    return matrnr


def WolfePowellSearch(f, x: np.array, d: np.array, sigma=1.0e-3, rho=1.0e-2, verbose=0):
    fx = f.objective(x) # store objective
    gradx = f.gradient(x) # store gradient
    descent = gradx.T @ d # store descent value

    if descent >= 0: # if not a descent direction
        raise TypeError('descent direction check failed!')

    if sigma <= 0 or sigma >= 0.5: # if sigma is out of range
        raise TypeError('range of sigma is wrong!')

    if rho <= sigma or rho >= 1: # if rho does not fit to sigma
        raise TypeError('range of rho is wrong!')

    if verbose: # print information
        print('Start WolfePowellSearch...') # print start

    t = 1 # initial step size guess

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    
    def W1(t):   # Define W1(t): checks Armijo condition for sufficient decrease
        return f.objective(x + t * d) <= fx + sigma * t * descent    # Return True if f(x + t*d) is sufficiently less than the linear approximation

    
    def W2(t):   # Define W2(t): checks curvature condition to ensure not too flat
        return f.gradient(x + t * d).T @ d >= rho * descent    # Return True if directional derivative at new point is large enough

    
    if not W1(t):    # Check if Armijo condition is violated
        t = t / 2     # Halve the step size t
        while not W1(t):   # Continue halving while Armijo condition fails
            t = t / 2      # Halve t again
        t_minus = t        # Store the last valid t as lower bound t₋
        t_plus = 2 * t     # Set upper bound t₊ to twice the valid t

    
    elif W2(t):         # If curvature condition is already satisfied
        return t        # Return current t since both conditions are satisfied

    else:                # If W1 is satisfied but W2 is not, enter fronttracking
        t = 2 * t        # Double the step size t
        while W1(t):     # As long as Armijo condition holds
            t = 2 * t    # Keep doubling t
        t_minus = t / 2  # Set lower bound t₋ to previous t before failure
        t_plus = t       # Set upper bound t₊ to current t

    
    t = t_minus          # Reset t to lower bound before refining

    
    while not W2(t):     # While curvature condition is not satisfied
        t = (t_minus + t_plus) / 2  # Set t to midpoint of current interval
        if W1(t):                    # If Armijo condition holds at new t
            t_minus = t              # Update lower bound t₋ to current t  
        else:                        # If Armijo condition fails
            t_plus = t               # Update upper bound t₊ to current t

    
    t = t_minus                      # Final output: set t to last valid lower bound
    
    # INCOMPLETE CODE ENDS
    if verbose: # print information
        xt = x + t * d # store solution point
        fxt = f.objective(xt) # get its objective
        gradxt = f.gradient(xt) # get its gradient
        print('WolfePowellSearch terminated with t=', t) # print terminatin and step size
        print('Wolfe-Powell: ', fxt, '<=', fx+t*sigma*descent, ' and ', gradxt.T @ d, '>=', rho*descent) # print Wolfe-Powell checks
    
    return t

