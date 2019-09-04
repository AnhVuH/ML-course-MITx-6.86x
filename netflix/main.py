import numpy as np
import kmeans
import common
import naive_em
from matplotlib import pyplot as plt
import em

X = np.loadtxt("toy_data.txt")

# TODO: Your code here
K = [1,2,3,4] # k=1,2,3,4
seed = [0,1,2,3,4] # k =0,1,2,3,4
def k_means_function(X,K,seed):
    init_model = common.init(X,K,seed)
    mixture, post, cost =kmeans.run(X,init_model[0],init_model[1])
    return mixture,post,cost



def naive_em_function(X,K,seed):
    init_model = common.init(X, K, seed)
    mixture, post, cost =naive_em.run(X,init_model[0],init_model[1])
    return mixture, post, cost

for i in range(len(K)):
    print("K=",K[i])
    for j in range(len(seed)):
        print("seed=",seed[j])
        # mixture1, post1, cost1 =k_means_function(X, K[i], seed[j])
        # print("K-mean :",cost1)


        mixture2, post2, cost2 =naive_em_function(X,K[i],seed[j])
        print("Naive EM :", cost2)
        common.bic(X, mixture2, cost2)
        # common.plot(X, mixture1, post1, "K-mean")
        # common.plot(X, mixture2, post2, "Naive EM")

