import numpy as np
import em
import common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")
X_gold_netflix = np.loadtxt("netflix_complete.txt")
X_netflix =np.loadtxt("netflix_incomplete.txt")

K = 12
n, d = X.shape
seed = [0,1,2,3,4]


# TODO: Your code here
for i in range(len(seed)):
    print(seed[i])
    init_model = common.init(X_netflix, K, seed[i])
    mixture, post, cost = em.run(X_netflix, init_model[0], init_model[1])
    X_pred = em.fill_matrix(X_netflix, mixture)
    rmse = common.rmse(X_gold_netflix,X_pred)
    print(cost)
    print(rmse)

# K= 4
# n,d = X.shape
# seed =0
# init_model = common.init(X, K, seed)
# mixture, post, cost = em.run(X, init_model[0], init_model[1])
# # print(mixture)
# X_pred = em.fill_matrix(X,mixture)
# print(X_pred)