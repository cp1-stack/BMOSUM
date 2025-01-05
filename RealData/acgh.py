import matplotlib
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
from sklearn.cluster import KMeans

time_start = time.time()

data = pd.read_csv(r'acghData1.csv', sep=',', header=None)
X = data.values.T.astype(np.float32)
n = np.shape(X)[1]
p = np.shape(X)[0]
k = int(0.05 * n)


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def BMOSUM(x):
    X = x
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
    return T_kn

aa=0
for j in range(1000):
    aa+=BMOSUM(X)
print(aa/1000)

time_end = time.time()
print('Time cost = %fs' % (time_end - time_start))
