"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n = X.shape[0]
    d = X.shape[1]
    k = mixture.p.shape[0]
    soft_count = np.ndarray((n, k))
    p_j_N = np.ndarray((n, k))
    for i in range(n):
        for j in range(k):
            p = mixture.p[j]
            mu = mixture.mu[j]
            var = mixture.var[j]
            N_mu_sigma = 1/(2* np.pi * var)**(d/2) * np.exp(-1/(2*var) * (np.linalg.norm(X[i]-mu))**2)
            p_j_N[i][j]= p * N_mu_sigma
    p_xi_theta = np.sum(p_j_N,axis =1)
    log_like = 0
    for i in range(n):
        soft_count[i] = p_j_N[i]/p_xi_theta[i]
        log_like += np.log(p_xi_theta[i])
    return soft_count , log_like


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n = X.shape[0]
    d = X.shape[1]
    k= post.shape[1]
    n_hat = np.sum(post, axis = 0)

    p_hat = n_hat/n
    mu_hat = np.ndarray((k,d))

    pj_i_xi = np.matmul(np.transpose(post),X)
    for j in range(k):
        mu_hat[j] = 1/n_hat[j] * pj_i_xi[j]

    var_hat = np.ndarray((k,))
    for j in range(k):
        pj_i_var =0
        for i in range(n):
            pj_i_var += post[i][j] * np.linalg.norm(X[i]-mu_hat[j])**2
        var_hat[j] = 1/(n_hat[j]*d) * pj_i_var
    mixture = GaussianMixture(p=p_hat, mu=mu_hat,var=var_hat)
    return mixture


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_cost = None
    cost = None
    while (prev_cost is None or (cost - prev_cost) > 1e-6*abs(cost)):

        prev_cost = cost
        post, cost = estep(X, mixture)
        mixture = mstep(X,post)

    return mixture,post,cost

