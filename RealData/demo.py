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

data = pd.read_csv(r'acghData.csv', sep=',', header=None)
X = data.values.T.astype(np.float32)
n = np.shape(X)[1]
p = np.shape(X)[0]
k = int(0.05 * n)


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def MBMOSUM(x):
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
    max2 = np.sort(Z_kn)[-1]
    # print(T_kn)
    # print(np.percentile(Z_kn, 99))

    cmap = get_cmap(43)
    # figure, ax = plt.subplots(2, 1)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=0.3)
    # ax = sns.heatmap(X)
    # plt.title('aCGH data')
    # ax[0].plot(X[1, :], color='red')
    # ax[0].set_title('The value of a dimension')

    # ax[1].plot(Z_kn, color='darkorange', ls='-', zorder=1)
    # ax[1].set_title('BMOSUM test')
    cp = []
    cp_value = []
    for j in Z_kn:
        if j > np.percentile(Z_kn, 95):
            cp.append(Z_kn.index(j))
            cp_value.append(j)
    print(cp)
    # plt.vlines(cp, 0, max2, color="r", ls='--', zorder=1)
    # plt.axhline(np.percentile(Z_kn, 99.9), color="purple")
    # plt.savefig("mosum_1000_95.png")
    # plt.show()

    # plt.scatter(cp, cp_value)
    # plt.show()
    X_real = X
    fig = plt.figure(figsize=(15, 10))
    matplotlib.rcParams['axes.unicode_minus'] = False
    ax1 = fig.add_subplot(211)
    ax = sns.heatmap(X_real)
    # plt.axhline(np.percentile(Z_kn, 100), color="purple")
    plt.title('aCGH data')

    CP = np.array(cp)
    CP_norm = (CP - CP.mean()) / CP.std()
    CP_value = np.array(cp_value)
    CP_value_norm = (CP_value - CP_value.mean()) / CP_value.std()
    X = np.array(list(zip(CP_norm, CP_value_norm)))
    X1 = CP_norm.reshape(-1, 1)
    X.reshape(-1, 1)
    # plt.scatter(cp, cp_value, marker='o')
    # plt.title('original data')
    # plt.sca(ax1)

    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    # std_scaler = StandardScaler()
    # std_scaler.fit(X)
    # X = std_scaler.transform(X)

    # ax2 = fig.add_subplot(212)
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.title('normalization')
    ax3 = fig.add_subplot(212)
    y_pred = DBSCAN(eps=0.05, min_samples=0.15).fit_predict(X1)
    plt.scatter(cp, cp_value, c=y_pred)

    data = np.vstack([y_pred, cp]).T.astype(int)
    np.savetxt("result.csv", data, fmt='%d',comments="", delimiter=",")
    data_cp = pd.read_csv('result.csv',names=['class','data'])
    # print(data_cp.head())
    mcp = data_cp.groupby('class').mean()
    print(mcp.iloc[:,0])
    np.savetxt("mcp.csv", mcp, fmt='%d', comments="", delimiter=",")

    # plt.vlines(mcp.iloc[:,0], 0, 10, color="g", ls='--', zorder=1)
    plt.title('DBSCAN cluster')
    plt.sca(ax3)
    # plt.savefig("aCGH.png")
    plt.show()

    fig = plt.figure(figsize=(15, 10))
    matplotlib.rcParams['axes.unicode_minus'] = False
    ax1 = fig.add_subplot(211)
    ax = sns.heatmap(X_real)
    plt.title('aCGH data')

    ax3 = fig.add_subplot(212)
    ax = sns.heatmap(X_real)
    plt.vlines(mcp.iloc[:, 0], 0, 43, color="yellow", ls='--', zorder=1)
    plt.title('aCGH data')
    # plt.savefig("aCGH_mcp.png")
    plt.show()

MBMOSUM(X)

time_end = time.time()
print('Time cost = %fs' % (time_end - time_start))
