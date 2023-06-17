import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import pandas
from sklearn.metrics import accuracy_score
import numpy.random as r
import struct
import random as rd
from threading import Thread
import os
import pandas as pd
from time import *
import random
import time
from scipy import stats
from scipy.stats import multivariate_t
import scipy.stats as st

time_start = time.time()
alpha = 95
n = 500
p = 20
delta_n = 0
delta = np.arange(0, 20, 1)
k = 25
mu = np.ones((p, 1))
M = np.zeros(p)
V_id = np.eye(p)
V_sd = 0.8 * np.ones(p) + 0.2 * np.eye(p)
m = 250
epoch = 200


def X(delta_n):
    # np.random.seed(t)

    def V_md():
        V_md = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                V_md[i, j] = 0.8 ** (abs(i - j))
        return V_md

    V_md = V_md()

    kesi = np.random.multivariate_normal(M, V_sd, n)
    kesi = kesi.T

    # rv = multivariate_t.rvs(shape=V_id, df=6, size=n)
    # kesi = rv.T

    mu = np.zeros((p, 1))

    def I_m(m):
        I_m = np.zeros((p, n))
        for i in range(n-m):
            I_m[:, i] = 1
        return I_m

    X = mu + delta_n * I_m(m) + kesi
    return X



def CUSUM(X):
    Z_kn = []
    for j in range(1,n-1):
        Z_j = ((1/j)*np.sum((X[:, : j]) ) - (1/(n-j))*np.sum((X[:, j:]) )) * np.sqrt((j*(n-j))/n)
        Z_j_inf = np.max(np.abs(Z_j))
        Z_kn.append(Z_j_inf)

    T_kn = np.argmax(Z_kn) + k
    # rmse = ((T_kn - m) / n) ** 2
    return np.max(Z_kn)

X11 = X(0)
quantile1 = []

for j in range(200):
    quantile1 = np.append(quantile1, CUSUM(X11))


T_quantile1 = np.percentile(quantile1, alpha)


# plt.hist(quantile)
# plt.show()

type2_1 = []


for dlt in delta:
    X11 = X(dlt)

    data1 = []

    for _ in range(epoch):
        data1.append(CUSUM(X11))

    num1 = sum(i <= T_quantile1 for i in data1) / epoch

    type2_1 = np.append(type2_1, num1)


type2 = [type2_1]
np.savetxt('PowerCUSUM.csv'.format(p,m), type2, delimiter=',', fmt='%4f')

time_end = time.time()
print('Time cost = %fs' % (time_end - time_start))
