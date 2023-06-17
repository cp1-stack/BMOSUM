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


time_start = time.time()
n = 200
p = 200
delta_n = 1
k = 15
mu = np.ones((p, 1))
M = np.zeros(p)
V_id = np.eye(p)
V_sd = 0.8 * np.ones(p) + 0.2 * np.eye(p)
epoch = 200

def BMOSUM(f):
    def X(t):
        np.random.seed(t)

        def V_md():
            V_md = np.zeros((p, p))
            for i in range(p):
                for j in range(p):
                    V_md[i, j] = 0.8 ** (abs(i - j))
            return V_md

        V_md = V_md()

        # kesi = np.random.multivariate_normal(M, V_id, n)
        # kesi = kesi.T
        rv = multivariate_t.rvs(shape=V_id, df=6, size=n)
        kesi = rv.T

        mu = np.ones((p, 1))

        def I_m(m):
            I_m = np.ones((p, n))
            for i in range(m):
                I_m[:, i] = 0
            return I_m

        X = mu + delta_n * I_m(m) + kesi
        return X

    X = X(f)
    Z_kn = []
    for j in range(k, n - k):
        x1k = np.mean(X[:, j - k: j])
        e1 = stats.norm.rvs(0, 1, size=(p, k))
        x2k = np.mean(X[:, j: j + k])
        e2 = stats.norm.rvs(0, 1, size=(p, k))
        Z_j = (np.sum((X[:, j - k: j] - x1k) * e1) - np.sum((X[:, j: j + k] - x2k) * e2)) / np.sqrt(2 * k)
        Z_j_inf = np.max(np.abs(Z_j))
        Z_kn.append(Z_j_inf)

    T_kn = np.argmax(Z_kn) + k
    rmse = ((T_kn - m) / n) ** 2
    return rmse


def BCUSUM(f):
    def X(t):
        np.random.seed(t)

        def V_md():
            V_md = np.zeros((p, p))
            for i in range(p):
                for j in range(p):
                    V_md[i, j] = 0.8 ** (abs(i - j))
            return V_md

        V_md = V_md()

        # kesi = np.random.multivariate_normal(M, V_id, n)
        # kesi = kesi.T
        rv = multivariate_t.rvs(shape=V_id, df=6, size=n)
        kesi = rv.T

        mu = np.ones((p, 1))

        def I_m(m):
            I_m = np.ones((p, n))
            for i in range(m):
                I_m[:, i] = 0
            return I_m

        X = mu + delta_n * I_m(m) + kesi
        return X

    X = X(f)
    Z_kn = []
    for j in range(1,n-1):
        e1 = stats.norm.rvs(0, 1, size=(p, j))
        e2 = stats.norm.rvs(0, 1, size=(p, n-j))
        Z_j = ((1/j)*np.sum((X[:, : j]) * e1) - (1/(n-j))*np.sum((X[:, j:]) * e2)) * np.sqrt((j*(n-j))/n)
        Z_j_inf = np.max(np.abs(Z_j))
        Z_kn.append(Z_j_inf)

    T_kn = np.argmax(Z_kn) + k
    rmse = ((T_kn - m) / n) ** 2
    return rmse



Rmse1 = []
Rmse2 = []

m = 100

for i in range(epoch):
    Rmse1.append(BMOSUM(i))
    Rmse2.append(BCUSUM(i))


n = 200
p = 400
delta_n = 1
k = 15
mu = np.ones((p, 1))
M = np.zeros(p)
V_id = np.eye(p)
V_sd = 0.8 * np.ones(p) + 0.2 * np.eye(p)

Rmse3 = []
Rmse4 = []

m = 100

for i in range(epoch):
    Rmse3.append(BMOSUM(i))
    Rmse4.append(BCUSUM(i))

n = 200
p = 600
delta_n = 1
k = 15
mu = np.ones((p, 1))
M = np.zeros(p)
V_id = np.eye(p)
V_sd = 0.8 * np.ones(p) + 0.2 * np.eye(p)

Rmse5 = []
Rmse6 = []

m = 100


for i in range(epoch):
    Rmse5.append(BMOSUM(i))
    Rmse6.append(BCUSUM(i))


RMSE = np.vstack([Rmse1, Rmse2, Rmse3,Rmse4,Rmse5,Rmse6])
# RMSE = np.vstack([Rmse1, Rmse2, Rmse3, Rmse4, Rmse5, Rmse6])

np.savetxt('rmse_200_id_t.csv', RMSE, delimiter=',', fmt='%4f')



time_end = time.time()
print('Time cost = %fs' % (time_end - time_start))
