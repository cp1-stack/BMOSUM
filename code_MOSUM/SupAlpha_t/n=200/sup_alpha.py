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
alpha = np.arange(0, 100, 10)
alpha1 = np.arange(0, 1, 0.1)
n = 200  # 200,500
p = 600  #200,400,600
delta_n = 0
k = 16  # 16,25
mu = np.ones((p, 1))
M = np.zeros(p)
V_id = np.eye(p)
V_sd = 0.8 * np.ones(p) + 0.2 * np.eye(p)
m = 0
epoch = 1000


def X():
    # np.random.seed(t)

    def V_md():
        V_md = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                V_md[i, j] = 0.8 ** (abs(i - j))
        return V_md

    V_md = V_md()
    #
    # kesi = np.random.multivariate_normal(M, V_md, n)  # id,md,sd
    # kesi = kesi.T

    rv = multivariate_t.rvs(shape=V_md, df=6, size=n)
    kesi = rv.T

    mu = np.ones((p, 1))

    def I_m(m):
        I_m = np.ones((p, n))
        for i in range(m):
            I_m[:, i] = 0
        return I_m

    X = mu + delta_n * I_m(m) + kesi
    return X


X = X()


def BCUSUM(X):
    Z_kn = []
    for j in range(1, n - 1):
        e1 = stats.norm.rvs(0, 1, size=(p, j))
        e2 = stats.norm.rvs(0, 1, size=(p, n - j))
        Z_j = ((1 / j) * np.sum((X[:, : j]) * e1) - (1 / (n - j)) * np.sum((X[:, j:]) * e2)) * np.sqrt(
            (j * (n - j)) / n)
        Z_j_inf = np.max(np.abs(Z_j))
        Z_kn.append(Z_j_inf)

    T_kn = np.argmax(Z_kn) + k
    # rmse = ((T_kn - m) / n) ** 2
    return T_kn


def BMOSUM(X):
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
    # rmse = ((T_kn - m) / n) ** 2
    return T_kn


quantile1 = []
quantile2 = []
for j in range(1000):
    quantile1 = np.append(quantile1, BMOSUM(X))
    quantile2 = np.append(quantile2, BCUSUM(X))

sup_alpha1 = []
sup_alpha2 = []
for a in alpha:
    data1 = []
    data2 = []
    T_quantile1 = np.percentile(quantile1, a)
    T_quantile2 = np.percentile(quantile2, a)
    for u in range(epoch):
        data1.append(BMOSUM(X))
        data2.append(BCUSUM(X))
    num1 = sum(i <= T_quantile1 for i in data1) / epoch
    num2 = sum(i <= T_quantile2 for i in data2) / epoch
    sup_alpha1 = np.append(sup_alpha1, num1)
    sup_alpha2 = np.append(sup_alpha2, num2)

data_sup = [abs(sup_alpha1-alpha1), abs(sup_alpha2-alpha1)]
np.savetxt('SupAlpha_p_{}_md_t.csv'.format(p), data_sup, delimiter=',', fmt='%4f')
time_end = time.time()
print('Time cost = %fs' % (time_end - time_start))

