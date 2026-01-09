import torch
import numpy as np
import pandas as pd

# code that simulates random paramter values
def simpars():
    nitems = 28
    # Simulate person parameters: multivariate standard normal


    # Simulate item parameters
    # a: discrimination parameters, only where Q == 1
    Q = pd.read_csv(f'./Qmatrices/Qmatrix_SE_simulation.csv', header=None).values[:nitems,:]
    theta = torch.randn(1000, Q.shape[1])
    a = ((torch.rand(nitems, Q.shape[1]) * 1.5 + 0.5) * Q).detach().numpy()

    # b: difficulty parameters
    b = (torch.rand(nitems) * 4 - 2).detach().numpy()  # Uniform[-2, 2]

    return a, b, theta, Q

# code that simulates data from a MIRT model
def simdata(a, b, theta):
    exponent = np.dot(theta, a.T) + b

    prob = np.exp(exponent) / (1 + np.exp(exponent))
    data = np.random.binomial(1, prob).astype(float)
    data = torch.Tensor(data)

    return data

# compute mean squared error
def MSE(est, true):
    """
    Mean square error
    Parameters
    ----------
    est: estimated parameter values
    true: true paremters values

    Returns
    -------
    the MSE
    """
    return np.mean(np.power(est-true,2))