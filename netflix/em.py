"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n = X.shape[0]
    k = mixture.p.shape[0]
    post = np.ndarray((n,k))

    f_u_j = np.ndarray((n,k))
    for u in range(n):
        X_cu_u = X[u, np.nonzero(X[u])]
        d = X_cu_u.shape[1]
        for j in range(k):
            p = mixture.p[j]
            mu_cu = mixture.mu[j,np.nonzero(X[u])]
            var = mixture.var[j]
            log_N_mu_sigma = -d/2 *np.log(2* np.pi * var) - (np.linalg.norm(X_cu_u-mu_cu))**2/(2*var)
            #log_N_mu_sigma = np.log(1/(2*np.pi*var)**(d/2)*np.exp(-(np.linalg.norm(X_cu_u-mu_cu))**2/(2*var)))
            f_u_j[u][j] = np.log(p +(1e-16)) + log_N_mu_sigma
    log_sum_exp = logsumexp(f_u_j, axis=1)
    log_like = np.sum(log_sum_exp, axis=0)

    for j in range(k):
        post[:,j]= np.exp(f_u_j[:,j]- log_sum_exp)

    return post, log_like

    # n= X.shape[0]
    # k = mixture.p.shape[0]
    # post = np.ndarray((n,k))
    # pj_N = np.ndarray((n,k))
    # f_u_j = np.ndarray((n,k))
    # for u in range(n):
    #     X_cu_u = X[u, np.nonzero(X[u])]
    #     d = X_cu_u.shape[1]
    #     for j in range(k):
    #         p = mixture.p[j]
    #         mu_cu = mixture.mu[j,np.nonzero(X[u])]
    #         var = mixture.var[j]
    #         N_mu_sigma = 1/(2* np.pi * var)**(d/2) * np.exp(-1/(2*var) * (np.linalg.norm(X_cu_u-mu_cu))**2)
    #         pj_N[u][j] = p * N_mu_sigma
    #         f_u_j[u][j] = np.log(p +(1e-16)) + np.log(N_mu_sigma)
    # p_xu_theta = np.sum(pj_N, axis=1)
    # for u in range(n):
    #     post[u] = pj_N[u]/p_xu_theta[u]
    #
    # log_sum_exp = logsumexp(f_u_j, axis= 1)
    # log_like = np.sum(log_sum_exp,axis =0)
    # print(post)
    # print(log_like)
    # return post, log_like

def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n,d = X.shape
    k = post.shape[1]
    p_hat = np.sum(post,axis =0)/n

    mu_hat = mixture.mu

    delta_l_cu = np.sign(X)
    delta_l_cu_pj_u = np.matmul(np.transpose(post),delta_l_cu)
    delta_l_cu_pj_u_x = np.matmul(np.transpose(post),X)

    for j in range(k):
        for l in range(d):
            if delta_l_cu_pj_u[j][l]>=1:
                mu_hat[j][l] = delta_l_cu_pj_u_x[j][l]/delta_l_cu_pj_u[j][l]

    var_hat = np.ndarray((k,))
    for j in range(k):
        sum_Cu_pj_u =0
        sum_pj_u_var =0
        for u in range(n):
            cu = np.nonzero(X[u])
            size_cu = np.size(cu)
            sum_Cu_pj_u += size_cu * post[u][j]
            sum_pj_u_var += post[u][j]*np.linalg.norm(X[u,cu]-mu_hat[j,cu])**2
        if sum_pj_u_var/sum_Cu_pj_u > min_variance:
            var_hat[j] = sum_pj_u_var/sum_Cu_pj_u
        else:
            var_hat[j] = min_variance

    return GaussianMixture(mu_hat,var_hat,p_hat)





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
        mixture = mstep(X,post,mixture)

    return mixture,post,cost

def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n,d = X.shape
    X_pred = np.ndarray((n,d))
    post, log_like = estep(X,mixture)
    for u in range(n):
        post_u = post[u]
        for l in range(d):
            if X[u][l] !=0:
                X_pred[u][l]= X[u][l]
            else:
                X_pred[u][l] = np.matmul(post_u,mixture.mu[:,l])

    return X_pred
