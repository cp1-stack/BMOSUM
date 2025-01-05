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
p = 200
delta_n = 0
delta = np.arange(0, 2, 0.25)
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

    kesi = np.random.multivariate_normal(M, V_id, n)
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



def BMOSUM(X):
    Z_kn = []
    for j in range(k, n - k):
        e1 = stats.norm.rvs(0, 1, size=(p, k))
        e2 = stats.norm.rvs(0, 1, size=(p, k))
        Z_j = (np.mean((X[:, j - k: j]) * e1) - np.mean((X[:, j: j + k]) * e2)) / np.sqrt(2 * k)
        Z_j_inf = np.max(np.abs(Z_j))
        Z_kn.append(Z_j_inf)

    T_kn = np.argmax(Z_kn) + k
    # rmse = ((T_kn - m) / n) ** 2
    return np.max(Z_kn)

def BCUSUM(X):
    Z_kn = []
    for j in range(1,n-1):
        e1 = stats.norm.rvs(0, 1, size=(p, j))
        e2 = stats.norm.rvs(0, 1, size=(p, n-j))
        Z_j = ((1/j)*np.sum((X[:, : j]) * e1) - (1/(n-j))*np.sum((X[:, j:]) * e2)) * np.sqrt((j*(n-j))/n)
        Z_j_inf = np.max(np.abs(Z_j))
        Z_kn.append(Z_j_inf)

    T_kn = np.argmax(Z_kn) + k
    # rmse = ((T_kn - m) / n) ** 2
    return np.max(Z_kn)


X11 = X(0)
quantile2 = []
quantile3 = []
for j in range(200):
    quantile2 = np.append(quantile2, BMOSUM(X11))
    quantile3 = np.append(quantile3, BCUSUM(X11))

T_quantile2 = np.percentile(quantile2, alpha)
T_quantile3 = np.percentile(quantile3, alpha)

# plt.hist(quantile)
# plt.show()

type2_2 = []
type2_3 = []

for dlt in delta:
    X11 = X(dlt)

    data2 = []
    data3 = []
    for _ in range(epoch):
        data2.append(BMOSUM(X11))
        data3.append(BCUSUM(X11))
    num2 = sum(i <= T_quantile2 for i in data2) / epoch
    num3 = sum(i <= T_quantile3 for i in data3) / epoch
    type2_2 = np.append(type2_2, num2)
    type2_3 = np.append(type2_3, num3)

type2 = [ type2_2, type2_3]
np.savetxt('Power_p_{}_id.csv'.format(p,m), type2, delimiter=',', fmt='%4f')

time_end = time.time()
print('Time cost = %fs' % (time_end - time_start))
