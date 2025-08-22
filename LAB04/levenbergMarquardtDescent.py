# Optimization for Engineers - Dr.Johannes Hild
# Levenberg-Marquardt descent

# Purpose: Find pmin to satisfy norm(jacobian_R.T @ R(pmin))<=eps

# Input Definition:
# R: error vector class with methods .residual() and .jacobian()
# p0: column vector in R**n (parameter point), starting point.
# eps: positive value, tolerance for termination. Default value: 1.0e-4.
# alpha0: positive value, starting value for damping. Default value: 1.0e-3.
# beta: positive value bigger than 1, scaling factor for alpha. Default value: 100.
# verbose: bool, if set to true, verbose information is displayed.

# Output Definition:
# pmin: column vector in R**n (parameter point)

# Required files:
# d = PrecCGSolver(A,b) from PrecCGSolver.py

# Test cases:
# p0 = np.array([[180],[0]], dtype=float)
# myObjective =  simpleValleyObjective(p0)
# xk = np.array([[[0], [0]], [[1], [2]]], dtype=float)
# fk = np.array([[2], [3]], dtype=float)
# myErrorVector = leastSquaresModel(myObjective, xk, fk)
# eps = 1.0e-4
# alpha0 = 1.0e-3
# beta = 100
# pmin = levenbergMarquardtDescent(myErrorVector, p0, eps, alpha0, beta, 1)
# should return pmin close to [[1], [1]]

import numpy as np
from PrecCGSolver import PrecCGSolver as PCG
from simpleValleyObjective import simpleValleyObjective
from leastSquaresFeasiblePoint import leastSquaresFeasiblePoint

def matrnr():
    # set your matriculation number here
    matrnr = 23533206
    return matrnr


def levenbergMarquardtDescent(R, p0: np.array, eps=1.0e-4, alpha0=1.0e-3, beta=100, verbose=0):
    if eps <= 0: # check for positive eps
        raise TypeError('range of eps is wrong!')

    if alpha0 <= 0: # check for positive alpha0
        raise TypeError('range of alpha0 is wrong!')

    if beta <= 1: # check for sufficiently large beta
        raise TypeError('range of beta is wrong!')

    if verbose: # print information
        print('Start levenbergMarquardtDescent...') # print start

    countIter = 0 # counter for loop iterations

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    
    p = p0.copy()                                        # initialize current iterate pₖ
    alpha = alpha0                                       # initialize damping parameter αₖ
    grad = R.jacobian(p).T @ R.residual(p)               # compute initial gradient norm
    # main loop: stop when ‖∇f‖ ≤ ε
    while np.linalg.norm(grad) > eps:
        J = R.jacobian(p)                                # evaluate Jacobian J(pₖ)
        r = R.residual(p)                                # evaluate residual R(pₖ)
        A = J.T @ J + alpha * np.eye(p.shape[0])         # form damped normal-equations matrix
        b = -J.T @ r                                     # form right-hand side -JᵀR
        d = PCG(A, b,delta=1.0e-6, verbose=0)            # solve A d = b via PreCG
        r_new = R.residual(p + d)                        # compute residual at trial point
        # check if the objective decreases
        if 0.5 * (r_new.T @ r_new) < 0.5 * (r.T @ r):
            p = p + d                                    # accept the step: pₖ₊₁ = pₖ + dₖ
            alpha = alpha0                               # reset damping to α₀
        else:
            alpha = beta * alpha                         # else increase damping αₖ ← β αₖ
        grad = R.jacobian(p).T @ R.residual(p)           # recompute gradient for new p
        countIter += 1                                   # increment iteration counter
    
    # INCOMPLETE CODE ENDS
    
    if verbose: # print information
        gradp = R.jacobian(p).T @ R.residual(p) # store final gradient
        print('levenbergMarquardtDescent terminated after ', countIter, ' steps with norm of gradient =', np.linalg.norm(gradp)) # print termination and gradient information

    return p

# p0 = np.array([[2], [-1]], dtype=float)
# myObjectives = np.array([simpleValleyObjective(p0)], dtype=object)
# myWeights = np.array([1], dtype=float)
# myErrorVector = leastSquaresFeasiblePoint(myObjectives, myWeights)
# x0 = np.array([[0], [4]], dtype=float)
# eps = 1.0e-6
# alpha0 = 1.0e-8
# beta = 1000
# xFeasible = levenbergMarquardtDescent(myErrorVector, x0, eps, alpha0, beta, 1)
# feasibleErrorVector = myErrorVector.residual(xFeasible)
# print("feasibleErrorVector",feasibleErrorVector)