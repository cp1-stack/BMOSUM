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

p = 200

delta = np.arange(0, 2, 0.25)
data1 = np.loadtxt(open("Power_p_{}_sd.csv".format(p), "rb"), delimiter=",")

# plt.figure(figsize=(10, 8), dpi=300)
# plt.tick_params(labelsize=20)

plt.rcParams.update({'font.size': 8})

plt.figure(1, figsize=(10, 8), dpi=300)

plt.subplot(231)
plt.plot(delta, 1 - data1[0, :], label='BMOSUM_200', color='deepskyblue', linestyle='--', marker='o')
plt.plot(delta, 1 - data1[1, :], label='BCUSUM_200', color='royalblue', linestyle='-.', marker='o')
plt.legend(loc=2)

plt.subplot(232)
plt.plot(delta, 1 - data1[2, :], label='BMOSUM_400', color='tomato', linestyle='--', marker='s')
plt.plot(delta, 1 - data1[3, :], label='BCUSUM_400', color='sandybrown', linestyle='-.', marker='s')
plt.legend(loc=2)

plt.subplot(233)
plt.plot(delta, 1 - data1[4, :], label='BMOSUM_600', color='gold', linestyle='--', marker='d')
plt.plot(delta, 1 - data1[5, :], label='BCUSUM_600', color='olive', linestyle='-.', marker='d')
plt.legend(loc=2)

plt.subplot(234)
plt.plot(delta, 1 - data1[6, :], label='BMOSUM_200', color='deepskyblue', linestyle='--', marker='o')
plt.plot(delta, 1 - data1[7, :], label='BCUSUM_200', color='royalblue', linestyle='-.', marker='o')
plt.legend(loc=2)

plt.subplot(235)
plt.plot(delta, 1 - data1[8, :], label='BMOSUM_400', color='tomato', linestyle='--', marker='s')
plt.plot(delta, 1 - data1[9, :], label='BCUSUM_400', color='sandybrown', linestyle='-.', marker='s')
plt.legend(loc=2)

plt.subplot(236)
plt.plot(delta, 1 - data1[10, :], label='BMOSUM_600', color='gold', linestyle='--', marker='d')
plt.plot(delta, 1 - data1[11, :], label='BCUSUM_600', color='olive', linestyle='-.', marker='d')
plt.legend(loc=2)

# plt.xlabel('Signal Size', fontsize=30)
# plt.ylabel('Power', fontsize=30)


plt.savefig("Power_sub.png".format(p))
plt.show()
