# Optimization for Engineers - Dr.Johannes Hild
# Preconditioned Conjugate Gradient Solver

# Purpose: PregCGSolver finds y such that norm(A * y - b) <= delta using incompleteCholesky as preconditioner

# Input Definition:
# A: real valued matrix nxn
# b: column vector in R ** n
# delta: positive value, tolerance for termination. Default value: 1.0e-6.
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# x: column vector in R ^ n(solution in domain space)

# Required files:
# L = incompleteCholesky(A, 1.0e-3, delta) from IncompleteCholesky.py
# y = LLTSolver(L, r) from LLTSolver.py

# Test cases:
# A = np.array([[4, 1, 0], [1, 7, 0], [ 0, 0, 3]], dtype=float)
# b = np.array([[5], [8], [3]], dtype=float)
# delta = 1.0e-6
# x = PrecCGSolver(A, b, delta, 1)
# should return x = [[1], [1], [1]]

# A = np.array([[484, 374, 286, 176, 88], [374, 458, 195, 84, 3], [286, 195, 462, -7, -6], [176, 84, -7, 453, -10], [88, 3, -6, -10, 443]], dtype=float)
# b = np.array([[1320], [773], [1192], [132], [1405]], dtype=float)
# delta = 1.0e-6
# x = PrecCGSolver(A, b, delta, 1)
# should return approx x = [[1], [0], [2], [0], [3]]


import numpy as np
from incompleteCholesky import incompleteCholesky as IC
from LLTSolver import LLTSolver as LLT


def matrnr():
    # set your matriculation number here
    matrnr = 23533206
    return matrnr


def PrecCGSolver(A: np.array, b: np.array, delta=1.0e-6, verbose=0):

    if verbose: # print information
        print('Start PrecCGSolver...') # print start

    countIter = 0 # counter for number of loop iterations
    
    L = IC(A) # initialize L as incomplete Cholesky decomposition of A
    x = LLT(L, b) # store solution of L x = b
    r = A @ x - b # residual of solving the linear system
    
    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    z = LLT(L, r)  # preconditioned residual z = LLT^{-1} r

    d = -z  # initial direction d = -z

    while np.linalg.norm(r) > delta: # loop until norm of residual is smaller than delta
        Ad = A @ d  # A * d
        rho = d.T @ Ad  # d^T * A * d

        t = (r.T @ z) / rho  # step size t

        x = x + t * d  # update solution

        r_old = r.copy()  # store old residual

        r = r + t * Ad  # update residual

        z_new = LLT(L, r)  # new preconditioned residual

        beta = (r.T @ z_new) / (r_old.T @ z)  # compute beta

        d = -z_new + beta * d  # update direction

        z = z_new  # update z

        countIter = countIter + 1  # increment iteration counter
    
    
    # INCOMPLETE CODE ENDS

    if verbose: # print information
        print('precCGSolver terminated after ', countIter, ' steps with norm of residual being ', np.linalg.norm(r)) # print termination

    return x

# A = np.array([[484, 374, 286, 176, 88], [374, 458, 195, 84, 3], [286, 195, 462, -7, -6], [176, 84, -7, 453, -10], [88, 3, -6, -10, 443]], dtype=float)
# b = np.array([[1320], [773], [1192], [132], [1405]], dtype=float)
# delta = 1.0e-6
# x = PrecCGSolver(A, b, delta, 1)
# print(x)