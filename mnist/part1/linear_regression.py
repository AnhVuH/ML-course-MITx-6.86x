import numpy as np
import scipy

from scipy import sparse

### Functions for you to fill in ###


#pragma: coderesponse template
def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        lambda_factor - the regularization constant (scalar)
    Returns:
        theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
        represents the y-axis intercept of the model and therefore X[0] = 1
    """
    # YOUR CODE HERE
    X_T = np.transpose(X)
    # X_T_X = np.matmul(X_T, X)
    # A = X_T_X + lambda_factor * np.identity(X_T_X.shape[0])
    # A_invertible = np.linalg.inv(A)
    # A_invert_X_T = np.matmul(A_invertible, X_T)
    # theta = np.matmul(A_invert_X_T, Y)

    X_T_sparse = scipy.sparse.csc_matrix(X_T)
    X_sparse =scipy.sparse.csc_matrix(X)
    X_T_X = X_T_sparse*X_sparse

    X_T_X = X_T_X.toarray()
    A = X_T_X + lambda_factor * np.identity(X_T_X.shape[0])
    A_invertible = np.linalg.inv(A)
    A_invert_sparse = scipy.sparse.csc_matrix(A_invertible)
    A_invert_X_T = A_invert_sparse*X_T_sparse
    A_invert_X_T = A_invert_X_T.toarray()

    return np.matmul(A_invert_X_T, Y)

    return theta
#pragma: coderesponse end

### Functions which are already complete, for you to use ###

def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)
