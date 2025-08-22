# Optimization for Engineers - Dr.Johannes Hild
# Newton descent

# Purpose: Find xmin to satisfy norm(gradf(xmin))<=eps
# Iteration: x_k = x_k + d_k
# d_k is the Newton direction

# Input Definition:
# f: objective class with methods .objective() and .gradient() and .hessian()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# d = PrecCGSolver(A,b) from PrecCGSolver.py

# Test cases:
# myObjective = bananaValleyObjective()
# x0 = np.array([[0], [1]])
# xmin = NewtonDescent(myObjective, x0, 1.0e-6, 1)
# should return
# xmin close to [[1],[1]]

import numpy as np
from PrecCGSolver import PrecCGSolver as PCG
from bananaValleyObjective import bananaValleyObjective 

def matrnr():
    # set your matriculation number here
    matrnr = 23533206
    return matrnr


def NewtonDescent(f, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0: # check for correct range of eps
        raise TypeError('range of eps is wrong!')

    if verbose: # print information
        print('Start NewtonDescent...') # print start

    countIter = 0 # counter for number of loop iterations
    xk = x0 # initialize with starting value

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    gradx = f.gradient(xk)# compute gradient at starting point
    
    while np.linalg.norm(gradx) > eps:# Loop until norm of gradient is less than epsilon
        
        Bk = f.hessian(xk)# Step 3a: set Bk to Hessian of f at xk

        dk = PCG(Bk, -gradx)# Step 3a: solve Bk * dk = -gradf(xk) using PrecCGSolver

        tk = 1.0# Step 3b: ignore line search, set tk = 1

        xk = xk + tk * dk# Step 3c: update xk

        gradx = f.gradient(xk)# Update gradient for the new xk

        countIter = countIter + 1# Increment iteration counter

    # INCOMPLETE CODE ENDS

    if verbose: # print information
        gradx = f.gradient(xk) # get gradient at solution
        print('NewtonDescent terminated after ', countIter, ' steps with norm of gradient =', np.linalg.norm(gradx)) # print termination and gradient norm information

    return xk
myObjective = bananaValleyObjective()
x0 = np.array([[0], [1]])
xk = NewtonDescent(myObjective, x0, 1.0e-6, 1)
print(xk)