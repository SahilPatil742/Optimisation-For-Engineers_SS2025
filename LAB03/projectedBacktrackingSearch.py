# Optimization for Engineers - Dr.Johannes Hild
# projected Wolfe-Powell line search

# Purpose: Find t to satisfy f(P(x+t*d))<f(x) + sigma*gradf(x).T@(P(x+t*d)-x) with P(x+t*d)-x being a descent direction
# and in addition but only if x+t*d is inside the feasible set gradf(x+t*d).T@d >= rho*gradf(x).T@d

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project()
# x: column vector in R ** n(domain point)
# d: column vector in R ** n(search direction)
# sigma: value in (0, 1 / 2), marks quality of decrease. Default value: 1.0e-3
# rho: value in (sigma, 1), marks quality of steepness. Default value: 1.0e-2
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# t: t is set to the biggest 2**m, such that 2**m satisfies the projected sufficient decrease condition
# and in addition if x+t*d is inside the feasible set, the second Wolfe-Powell condition holds

# Required files:
# <none>

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[-2], [1]])
# b = np.array([[2], [2]])
# eps = 1.0e-6
# myBox = projectionInBox(a, b, eps)
# x = np.array([[1], [1]])
# d = np.array([[-1.99], [0]])
# sigma = 0.5
# rho = 0.75
# t = projectedBacktrackingSearch(myObjective, myBox, x, d, sigma, rho, 1)
# should return t = 0.5

import numpy as np
from simpleValleyObjective import simpleValleyObjective
from projectionInBox import projectionInBox

def matrnr():
    # set your matriculation number here
    matrnr = 23533206
    return matrnr


def projectedBacktrackingSearch(f, P, x: np.array, d: np.array, sigma=1.0e-4, rho=1.0e-2, verbose=0):
    xp = P.project(x) # initialize with projected starting point
    fx = f.objective(xp) # get current objective
    gradx = f.gradient(xp) # get current gradient
    descent = gradx.T @ d # descent direction check value

    if descent >= 0: # if not a descent direction
        raise TypeError('descent direction check failed!')

    if sigma <= 0 or sigma >= 0.5: # if sigma is out of range
        raise TypeError('range of sigma is wrong!')

    if rho <= sigma or rho >= 1: # if rho does not fit to sigma
        raise TypeError('range of rho is wrong!')

    if verbose: # print information
        print('Start projectedBacktracking...') # print start

    t = 1 # starting guess for t

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
        
    def W1(t):   # Define W1(t) according to the Wolfe-Powell Armijo condition and descent direction
        xt = P.project(x + t * d)  # project the candidate point x + t * d
        return (gradx.T @ (xt - xp) < 0) and (f.objective(xt) <= fx + sigma * gradx.T @ (xt - xp))  # Armijo + descent

    def W2(t):  # Define W2(t) according to the curvature condition or if projection changed
        xt = P.project(x + t * d)  # project the candidate point
        return (not np.array_equiv(xt, x + t * d)) or (f.gradient(xt).T @ d >= rho * descent)  # curvature or projection change

    if not W1(t):  # backtracking if W1 fails
        t = t / 2  # halve t
        while not W1(t):  # repeat while W1 still fails
            t = t / 2  # halve t again
        t_minus = t  # set lower bracket
        t_plus = 2 * t  # set upper bracket
    elif W2(t):  # if W1 is okay and W2 satisfied
        t_minus = t  # accept t as final step size
    else:
        t = 2 * t  # fronttracking: double t
        while W1(t) and np.array_equiv(P.project(x + t * d), x + t * d):  # check W1 holds and projection didn't change
            t = 2 * t  # keep doubling
        t_plus = t  # set upper bracket
        t_minus = t / 2  # set lower bracket

    t = t_minus  # initialize refinement from lower bracket

    while not W2(t):  # refine until W2 satisfied
        t = (t_minus + t_plus) / 2  # bisection
        if W1(t):  # if W1 holds
            t_minus = t # update t_minus
        else:
            t_plus = t  # otherwise update t_plus
    
    # INCOMPLETE CODE ENDS

    if verbose: # print verbose information
        xt = P.project(x + t * d) # get x+td for found step size t
        fxt = f.objective(xt) # get objective value at this point
        print('projectedBacktracking terminated with t=', t) # print termination
        print('Sufficient decrease: ', fxt, '<=', fx+t*sigma*descent) # print result of sufficient decrease check

    return t

# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[-2], [1]])
# b = np.array([[2], [2]])
# eps = 1.0e-6
# myBox = projectionInBox(a, b, eps)
# x = np.array([[1], [1]])
# d = np.array([[-1.99], [0]])
# sigma = 0.499
# rho = 0.75
# t = projectedBacktrackingSearch(myObjective, myBox, x, d, sigma, rho, 1)
# print(f't={t}')