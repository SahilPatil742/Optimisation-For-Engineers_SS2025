# Optimization for Engineers - Dr.Johannes Hild
# Least squares feasible point

# Purpose: Provides .residual() and .jacobian() of the least squares mapping x -> 0.5*sum_k (p_k*h_k(x))**2

# Input Definition:
# hArray: N-dimensional array with objective classes mapping R**n->R with methods .objective() and .gradient(), equality constraints
# p: column vector in R**N, weights for the constraints

# Output Definition:
# residual(): column vector in R**N, the k-th entry is p[k]*h[k](x)
# jacobian(): matrix in R**Nxm, the [k,j]-th entry returns: partial derivative with respect to x_j of (p[k]*h[k](x))

# Required files:
# <none>

# Test cases:
# p0 = np.array([[2],[-1]], dtype=float)
# myObjectives =  np.array([simpleValleyObjective(p0)], dtype=object)
# myWeights = np.array([1], dtype=float)
# myErrorVector = leastSquaresFeasiblePoint(myObjectives, myWeights)
# x0 = np.array([[0],[4]], dtype=float)
# should return
# myErrorVector.residual(x0) close to [[18]]
# myErrorVector.jacobian(x0) = [[0, 12]]

import numpy as np
from simpleValleyObjective import simpleValleyObjective

def matrnr():
    # set your matriculation number here
    matrnr = 23533206
    return matrnr


class leastSquaresFeasiblePoint:

    def __init__(self, hArray:np.array, p: np.array):
        self.hArray = hArray # array storing all constraints
        self.p = p # weights for the constraints
        self.N = hArray.shape[0] # number of constraints

    def residual(self, x: np.array):
        myResidual = np.zeros((self.N, 1)) # initialize residual vector as zero vector

        # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
        for k in range(self.N):  # iterate through all constraints
            hk = self.hArray[k].objective(x)  # evaluate constraint k at point x
            myResidual[k, 0] = self.p[k] * hk  # scale by weight and assign to residual vector

        # INCOMPLETE CODE ENDS

        return myResidual

    def jacobian(self, x: np.array):
        myJacobian = np.zeros((self.N, x.shape[0])) # initialize jacobian matrix as zero matrix

        # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
        for k in range(self.N):  # iterate through all constraints
            hk_grad = self.hArray[k].gradient(x)  # evaluate gradient of constraint k at point x
            myJacobian[k, :] = self.p[k] * hk_grad.T  # scale by weight and assign transpose of gradient to jacobian matrix row
        # INCOMPLETE CODE ENDS

        return myJacobian
    
# p0 = np.array([[2],[-1]], dtype=float)
# myObjectives =  np.array([simpleValleyObjective(p0)], dtype=object)
# myWeights = np.array([1], dtype=float)
# myErrorVector = leastSquaresFeasiblePoint(myObjectives, myWeights)
# x0 = np.array([[0],[4]], dtype=float)
# residual = myErrorVector.residual(x0)
# jacobian = myErrorVector.jacobian(x0)
# print("Residual:", residual)
# print("Jacobian:", jacobian)

