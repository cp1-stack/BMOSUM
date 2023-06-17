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

time_start = time.time()
n = 1000
p = 100
delta_n = 5
k = 20
mu = np.ones((p, 1))
M = np.zeros(p)
V_id = np.eye(p)
V_sd = 0.8 * np.ones(p) + 0.2 * np.eye(p)


def MOSUM():
    def X(t):
        np.random.seed(t)

        def V_md():
            V_md = np.zeros((p, p))
            for i in range(p):
                for j in range(p):
                    V_md[i, j] = 0.8 ** (abs(i - j))
            return V_md

        V_md = V_md()

        kesi = np.random.multivariate_normal(M, V_id, n)
        kesi = kesi.T

        mu = np.ones((p, 1))

        def I_m(m):
            I_m = np.ones((p, n))
            for i in range(m):
                I_m[:, i] = 0
            return I_m

        X = mu + delta_n * I_m(m) + kesi
        return X

    X = X(random.randint(1, 200))
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
    mae = np.abs((T_kn - m) / n)
    rmse = ((T_kn - m) / n) ** 2
    return [rmse, mae]


Rmse1 = []
Mse1 = []
Mae1 = []
for t in range(200, 1000, 100):
    m = t
    a = 0
    b = 0
    for i in range(100):
        a = a + MOSUM()[0]
        b = b + MOSUM()[1]

    rmse = np.sqrt(a / 100)
    mse = a / 100
    mae = b / 100

    Rmse1.append(rmse)
    Mse1.append(mse)
    Mae1.append(mae)

time_end = time.time()
print('Time cost = %fs' % (time_end - time_start))

np.savetxt('100_1000_20_rmse_mse_mae.txt', (Rmse1, Mse1, Mae1), delimiter=',', fmt='%4f')

k = 30
Rmse2 = []
Mse2 = []
Mae2 = []
for t in range(200, 1000, 100):
    m = t
    a = 0
    b = 0
    for i in range(100):
        a = a + MOSUM()[0]
        b = b + MOSUM()[1]

    rmse = np.sqrt(a / 100)
    mse = a / 100
    mae = b / 100

    Rmse2.append(rmse)
    Mse2.append(mse)
    Mae2.append(mae)

time_end = time.time()
print('Time cost = %fs' % (time_end - time_start))

np.savetxt('100_1000_30_rmse_mse_mae.txt', (Rmse2, Mse2, Mae2), delimiter=',', fmt='%4f')

k = 50
Rmse3 = []
Mse3 = []
Mae3 = []
for t in range(200, 1000, 100):
    m = t
    a = 0
    b = 0
    for i in range(100):
        a = a + MOSUM()[0]
        b = b + MOSUM()[1]

    rmse = np.sqrt(a / 100)
    mse = a / 100
    mae = b / 100

    Rmse3.append(rmse)
    Mse3.append(mse)
    Mae3.append(mae)

time_end = time.time()
print('Time cost = %fs' % (time_end - time_start))

np.savetxt('100_1000_50_rmse_mse_mae.txt', (Rmse3, Mse3, Mae3), delimiter=',', fmt='%4f')

k = 100
Rmse4 = []
Mse4 = []
Mae4 = []
for t in range(200, 1000, 100):
    m = t
    a = 0
    b = 0
    for i in range(100):
        a = a + MOSUM()[0]
        b = b + MOSUM()[1]

    rmse = np.sqrt(a / 100)
    mse = a / 100
    mae = b / 100

    Rmse4.append(rmse)
    Mse4.append(mse)
    Mae4.append(mae)

time_end = time.time()
print('Time cost = %fs' % (time_end - time_start))

np.savetxt('100_1000_100_rmse_mse_mae.txt', (Rmse4, Mse4, Mae4), delimiter=',', fmt='%4f')
