"""
Created on Wed Sep  2 10:36:41 2020.

tc_algos.py

Tensor Completion Algorithms:
1. Simple Low Rank Tensor Completion aka SiLRTC
2. High accurracy Low Rank Tensor Completion aka HalRTC
3.

Based on code developed by Lior Bragilevsky (Multimedia Lab, Simon Fraser University).
SiLRTC-complete.py
The code has been modified to run with DFTS re-packetized tensors.

Ref:
1. L. Bragilevsky and I. V.Bajić, “Tensor completion methods for collaborative
intelligence,” IEEE Access, vol. 8, pp. 41162–41174, 2020.

"""
# Libraries for tensor completion methods.
import numpy as np
import copy
from .tensorly_base import *
"""
If you able to install tensorly, you can import tensorly directly and comment out
the line above. If you are running your experiment on a ComputeCanada cluster,
you won't be able to install tensorly.
"""
# --------------------------------------------------------------------------- #
# General functions used by tensor completion methods.

def swap(a, b):
    """
    Swap a and b.

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.

    Returns
    -------
    b : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.

    """

    return b, a

def makeOmegaSet_rowPacket(p, n, sizeOmega):
    """


    Parameters
    ----------
    p : TYPE
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    sizeOmega : TYPE
        DESCRIPTION.

    Returns
    -------
    subs : TYPE
        DESCRIPTION.

    """
    if sizeOmega > np.prod(n):
        print("OmegaSet size is too high, requested size of Omega is bigger than the tensor itself!")

    row_omega = int(np.ceil(p * n[0] * n[-1]))
    idx = np.random.randint(low=1, high=n[0]*n[-1], size=(row_omega, 1))
    Omega = np.unique(idx)

    while len(Omega) < row_omega:
        Omega = np.reshape(Omega, (len(Omega), 1))
        temp = np.random.randint(low=1, high=n[0]*n[-1], size=(row_omega-len(Omega), 1))
        idx = np.concatenate((Omega, temp), axis=0)
        Omega = np.unique(idx)

    Omega = np.sort(Omega[0:row_omega])
    subs = np.unravel_index(Omega, [n[0],n[-1]])

    temp = np.ones((1, np.array(subs).shape[-1]))

    # Create the columns (for each permutation increment by 1 so that all the columns get covered for each row)
    res = []
    for i in range(n[1]):
        concat = np.concatenate((subs, i*temp), axis=0)
        res.append(concat)

    # Swap axis 2 and 3 to be correct
    swap_res = []
    for i in range(len(res)):
        a = res[i][0]
        b = res[i][1]
        c = res[i][2]
        b, c = swap(b, c)

        swapped_order = np.stack((a,b,c))
        swap_res.append(swapped_order)

    # Concatenating the list formed above to give a 3 by x matrix
    subs = swap_res[0]
    for i in range(len(swap_res)-1):
        subs = np.concatenate((subs, swap_res[i+1]), axis=1)

    return subs

def ReplaceInd(X, subs, vals):
    """
    Replace in X values given by vals in location given by subs.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    subs : TYPE
        DESCRIPTION.
    vals : TYPE
        DESCRIPTION.

    Returns
    -------
    X : TYPE
        DESCRIPTION.

    """
    for j in range(len(vals)):
        x, y, z = subs[j,0], subs[j,1], subs[j,2]
        X[x, y, z] = vals[j]

    return X

def shrinkage(X, t):
	"""
    Perform shrinkage with threshold t on matrix X.

    Refer to Bragilevsky, Bajic paper for explanation.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.

    Returns
    -------
    d : TYPE
        DESCRIPTION.
    """
    # SVD decomposition with an offset (s needs to be made diagonal)
	u, s, v = np.linalg.svd(X, full_matrices=False) # matlab uses 'econ' to not produce full matricies
	s = np.diag(s)

	for i in range(s.shape[0]):
		s[i,i]=np.max(s[i,i]-t, 0)

	# reconstructed matrix
	d =  np.matmul(np.matmul(u, s), v)

	return d
# --------------------------------------------------------------------------- #
# Simple Low Rank Tensor Completion method.
# Adapted from Lior's code.

def fn_silrtc_demo(image,num_iters_K,p):
    """
    Demonstrate operation of SILRTC with random loss.

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.
    K : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.

    Returns
    -------
    X_estimated : TYPE
        DESCRIPTION.

    """
    n = list(image.shape)

    subs = np.transpose(np.array(makeOmegaSet_rowPacket(p, n, np.uint32(np.round(p*np.prod(n))))))
    subs = np.array(subs, dtype=np.uint32)

    vals = list(map(lambda x, y, z: image[x][y][z], subs[:,0], subs[:,1], subs[:,2]))
    X_corrupted = np.zeros(n)
    X_corrupted = ReplaceInd(X_corrupted, subs, vals)

    X_estimated = X_corrupted

    b = np.abs(np.random.randn(3,1))/200
    a = np.abs(np.random.randn(3,1))
    a = a/np.sum(a)

    for q in range(num_iters_K):
        M = np.zeros(image.shape)
        for i in range(3):
            svd = shrinkage(unfold(X_estimated,i), a[i]/b[i])
            M = M + fold(b[i]*svd, i, image.shape)

        M = M/np.sum(b)
        # Update both M & X as they are used in the next cycle
        M = ReplaceInd(M, subs, vals)
        X_estimated = M
    return X_estimated


def fn_silrtc_damaged(X_corrupted,num_iters_K,subs,vals):
    """
    Perform SiLRTC on damaged tensors.

    Parameters
    ----------
    X_corrupted : TYPE
        DESCRIPTION.
    num_iters_K : TYPE
        DESCRIPTION.
    subs : TYPE
        DESCRIPTION.
    vals : TYPE
        DESCRIPTION.

    Returns
    -------
    X_estimated : TYPE
        DESCRIPTION.

    """
    X_estimated = X_corrupted

    b = np.abs(np.random.randn(3,1))/200
    a = np.abs(np.random.randn(3,1))
    a = a/np.sum(a)

    for q in range(num_iters_K):
        #print(f"SilRTC iteration {q}")
        M = np.zeros(X_corrupted.shape)
        for i in range(3):
            svd = shrinkage(unfold(X_estimated,i), a[i]/b[i])
            M = M + fold(b[i]*svd, i, X_corrupted.shape)

        M = M/np.sum(b)
        # Update both M & X as they are used in the next cycle
        M = ReplaceInd(M, subs, vals)
        X_estimated = M
    return X_estimated

def fn_silrtc_damaged_error(X_corrupted,num_iters_K,subs,vals):
    """
    Perform SiLRTC on damaged tensors. Keep track of error.

    Parameters
    ----------
    X_corrupted : TYPE
        DESCRIPTION.
    num_iters_K : TYPE
        DESCRIPTION.
    subs : TYPE
        DESCRIPTION.
    vals : TYPE
        DESCRIPTION.

    Returns
    -------
    X_estimated : TYPE
        DESCRIPTION.
    error_iters:

    """
    X_estimated = copy.deepcopy(X_corrupted)

    b = np.abs(np.random.randn(3,1))/200
    a = np.abs(np.random.randn(3,1))
    a = a/np.sum(a)

    error_iters = np.zeros([num_iters_K],dtype=np.float64)
    X_estimated_prev = np.zeros_like(X_estimated)

    row, col, dep = X_corrupted.shape
    ArrSize_iters = (row,col,dep,num_iters_K)
    X_estimated_iters = np.zeros(ArrSize_iters)

    for q in range(num_iters_K):
        #print(f"SilRTC iteration {q}")
        M = np.zeros(X_corrupted.shape)
        for i in range(3):
            svd = shrinkage(unfold(X_estimated,i), a[i]/b[i])
            M = M + fold(b[i]*svd, i, X_corrupted.shape)

        M = M/np.sum(b)
        # Update both M & X as they are used in the next cycle
        X_estimated = ReplaceInd(M, subs, vals)

        error_iters[q] = np.sqrt(np.sum(np.square(np.subtract(X_estimated,X_estimated_prev))))
        X_estimated_prev = X_estimated

        X_estimated_iters[:,:,:,q] = X_estimated


    return X_estimated_iters, error_iters
# --------------------------------------------------------------------------- #
# High accuracy Low Rank Tensor Completion.
# Adapted from Lior's code.

def fn_halrtc_damaged(X_corrupted,num_iters_K,subs,vals):
    """
    Perform HaLRTC on damaged tensors.

    Parameters
    ----------
    X_corrupted : TYPE
        DESCRIPTION.
    num_iters_K : TYPE
        DESCRIPTION.
    subs : TYPE
        DESCRIPTION.
    vals : TYPE
        DESCRIPTION.

    Returns
    -------
    X_estimated : TYPE
        DESCRIPTION.

    """
    X_estimated = np.copy(X_corrupted)

    a = np.abs(np.random.randn(3,1))
    a = a/np.sum(a)
    rho = 1e-6

    # Create tensor holders for Mi and Yi done to simplify variable storage
    row, col, dep = X_corrupted.shape
    ArrSize = (row, col, dep, X_corrupted.ndim)

    Mi = np.zeros(ArrSize)
    Yi = np.zeros(ArrSize)

    for q in range(num_iters_K):
        #print(f"HalRTC iteration {q}")
        # Calculate Mi tensors (Step 1)
        for i in range(3):
            temp = unfold(X_estimated,i) + (unfold(np.squeeze(Yi[:,:,:,i]),i)/rho)
            Mi[:,:,:,i] = fold(shrinkage(temp, a[i]/rho), i, X_corrupted.shape)
        # Update X (Step 2)
        X_est = np.sum(Mi-(Yi/rho),axis=3)/3
        X_estimated = ReplaceInd(X_est, subs, vals)

        # Update Yi tensors (Step 3)
        for i in range(ArrSize[-1]):
            Yi[:,:,:,i] = np.squeeze(Yi[:,:,:,i])-rho*(np.squeeze(Mi[:,:,:,i])-X_estimated)

        # Modify rho to help convergence
        rho = 1.2*rho

    return X_estimated

def fn_halrtc_damaged_error(X_corrupted,num_iters_K,subs,vals):
    """
    Perform HaLRTC on damaged tensors.

    Parameters
    ----------
    X_corrupted : TYPE
        DESCRIPTION.
    num_iters_K : TYPE
        DESCRIPTION.
    subs : TYPE
        DESCRIPTION.
    vals : TYPE
        DESCRIPTION.

    Returns
    -------
    X_estimated : TYPE
        DESCRIPTION.

    """
    X_estimated = np.copy(X_corrupted) #copy.deepcopy(X_corrupted)

    a = np.abs(np.random.randn(3,1))
    a = a/np.sum(a)
    rho = 1e-6

    # Create tensor holders for Mi and Yi done to simplify variable storage
    row, col, dep = X_corrupted.shape
    ArrSize = (row, col, dep, X_corrupted.ndim)

    Mi = np.zeros(ArrSize)
    Yi = np.zeros(ArrSize)

    error_iters = np.zeros([num_iters_K],dtype=np.float64)
    X_estimated_prev = np.zeros_like(X_estimated)

    ArrSize_iters = (row,col,dep,num_iters_K)
    X_estimated_iters = np.zeros(ArrSize_iters)

    for q in range(num_iters_K):
        #print(f"HalRTC iteration {q}")
        # Calculate Mi tensors (Step 1)
        for i in range(3):
            temp = unfold(X_estimated,i) + (unfold(np.squeeze(Yi[:,:,:,i]),i)/rho)
            Mi[:,:,:,i] = fold(shrinkage(temp, a[i]/rho), i, X_corrupted.shape)
        # Update X (Step 2)
        X_est = np.sum(Mi-(Yi/rho),axis=3)/3
        X_estimated = ReplaceInd(X_est, subs, vals)
        X_estimated_iters[:,:,:,q] = X_estimated

        # Update Yi tensors (Step 3)
        for i in range(ArrSize[-1]):
            Yi[:,:,:,i] = np.squeeze(Yi[:,:,:,i])-rho*(np.squeeze(Mi[:,:,:,i])-X_estimated)

        # Modify rho to help convergence
        rho = 1.2*rho

        error_iters[q] = np.sqrt(np.sum(np.square(np.subtract(X_estimated,X_estimated_prev))))
        X_estimated_prev = X_estimated

    return X_estimated_iters, error_iters
