# Optimization for Engineers - Dr.Johannes Hild
# projected BFGS descent

# Purpose: Find xmin to satisfy norm(xmin - P(xmin - gradf(xmin)))<=eps
# Iteration: x_k = P(x_k + t_k * d_k)
# d_k is the reduced BFGS direction. If a descent direction check fails, d_k is set to steepest descent and the BFGS matrix is reset.
# t_k results from projected backtracking

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# d = PrecCGSolver(A,b) from PrecCGSolver.py
# t = projectedBacktrackingSearch(f, P, x, d) from projectedBacktrackingSearch.py

# Test cases:
# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

import numpy as np
from projectedBacktrackingSearch import projectedBacktrackingSearch as PB
from PrecCGSolver import PrecCGSolver as PG
from simpleValleyObjective import simpleValleyObjective
from projectionInBox import projectionInBox

def matrnr():
    # set your matriculation number here
    matrnr = 23533206
    return matrnr


def projectedBFGSDescent(f, P, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0: # check for positive eps
        raise TypeError('range of eps is wrong!')

    if verbose: # print information
        print('Start projectedBFGSDescent...') # print start

    countIter = 0 # counter for number of loop iterations
    xp = P.project(x0) # initialize with projected starting point

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    
    xk = xp.copy()  # initialize current iterate
    n = xk.shape[0]  # get dimension
    Hk = np.eye(n)  # initialize Hessian approximation as identity
    Ak = P.activeIndexSet(xk)  # get initial active set

    gradx = f.gradient(xk)  # gradient at xk
    pg = xk - P.project(xk - gradx)  # projected gradient

    while np.linalg.norm(pg) >= eps:  # stationarity condition
        countIter += 1  # increment iteration count

        dk = PG(Hk, -gradx)  # solve Hk * dk = -gradx

        if gradx.T @ dk >= 0:  # if not descent direction
            dk = -gradx  # fallback to steepest descent
            Hk = np.eye(n)  # reset Hessian

        tk = PB(f, P, xk, dk)  # compute step size using projected backtracking
        xplus = P.project(xk + tk * dk)  # take projected step
        Aplus = P.activeIndexSet(xplus)  # get new active set

        if not np.array_equal(Aplus, Ak):  # active set changed
            Hk[Aplus, :] = np.eye(n)[Aplus, :]  # reduce rows
            Hk[:, Aplus] = np.eye(n)[:, Aplus]  # reduce cols
        else:
            gradplus = f.gradient(xplus)  # gradient at new point
            sk = xplus - xk  # Δx_k
            yk = gradplus - gradx  # Δg_k

            yTs = (yk.T @ sk).item()  # scalar Δg_kᵀ Δx_k
            if yTs <= eps**2:  # curvature condition failed
                Hk = np.eye(n)  # reset Hessian
            else:
                Hs = Hk @ sk  # compute Hk Δx_k
                sTHs = (sk.T @ Hs).item()  # scalar Δx_kᵀ Hk Δx_k

                Hk = Hk + (yk @ yk.T) / yTs - (Hs @ Hs.T) / sTHs # Apply BFGS update (Lemma 6.6, Eq. 6.9)

                Hk[Aplus, :] = np.eye(n)[Aplus, :]  # reduce rows
                Hk[:, Aplus] = np.eye(n)[:, Aplus]  # reduce cols

        xk = xplus  # update current point
        Ak = Aplus  # update active set
        gradx = f.gradient(xk)  # update gradient
        pg = xk - P.project(xk - gradx)  # update projected gradient

    xp = xk  # assign final result


    # INCOMPLETE CODE ENDS

    if verbose: # print information
        gradx = f.gradient(xp) # get gradient
        stationarity = np.linalg.norm(xp - P.project(xp - gradx)) # get stationarity
        print('projectedBFGSDescent terminated after ', countIter, ' steps with stationarity =', np.linalg.norm(stationarity)) # print termination

    return xp

# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# print("xmin=", xmin)