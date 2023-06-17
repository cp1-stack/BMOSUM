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
alpha = np.arange(0.1, 1, 0.1)
n = 10
p = 1
delta_n = 0
k = 2
mu = np.ones((p, 1))
M = np.zeros(p)
V_id = np.eye(p)
V_sd = 0.8 * np.ones(p) + 0.2 * np.eye(p)
m =0


# def X():
#     # np.random.seed(t)
#
#     def V_md():
#         V_md = np.zeros((p, p))
#         for i in range(p):
#             for j in range(p):
#                 V_md[i, j] = 0.8 ** (abs(i - j))
#         return V_md
#
#     V_md = V_md()
#
#     kesi = np.random.multivariate_normal(M, V_id, n)
#     kesi = kesi.T
#
#     mu = np.ones((p, 1))
#
#     def I_m(m):
#         I_m = np.ones((p, n))
#         for i in range(m):
#             I_m[:, i] = 0
#         return I_m
#
#     X = mu + delta_n * I_m(m) + kesi
#     return X
#
#
# X = X()
#
# def BMOSUM():
#
#     Z_kn = []
#     for j in range(k, n - k):
#         x1k = np.mean(X[:, j - k: j])
#         e1 = stats.norm.rvs(0, 1, size=(p, k))
#         x2k = np.mean(X[:, j: j + k])
#         e2 = stats.norm.rvs(0, 1, size=(p, k))
#         Z_j = (np.sum((X[:, j - k: j] - x1k) * e1) - np.sum((X[:, j: j + k] - x2k) * e2)) / np.sqrt(2 * k)
#         Z_j_inf = np.max(np.abs(Z_j))
#         Z_kn.append(Z_j_inf)
#
#     T_kn = np.argmax(Z_kn) + k
#     rmse = ((T_kn - m) / n) ** 2
#     return T_kn

delta = [0,0.1,0.2]
for a in delta:
    print(a)