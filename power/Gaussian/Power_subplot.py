import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
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

delta = np.arange(0, 2, 0.25)
data1 = np.loadtxt(open("plot_power_sd.csv", "rb"), delimiter=",")

# plt.figure(figsize=(10, 8), dpi=300)
# plt.tick_params(labelsize=20)

plt.rcParams.update({'font.size': 8})
matplotlib.rcParams['text.usetex'] = True
fig = plt.figure(1, figsize=(14, 6), dpi=300)

plt.subplot(161)
plt.plot(delta, 1 - data1[0, :], label='Our Method', color='deepskyblue', linestyle='--', marker='o')
plt.plot(delta, 1 - data1[1, :], label='BCUSUM', color='royalblue', linestyle='-.', marker='o')
plt.legend(loc=2)
plt.title('n=200, $p_{n}=200$')
plt.subplot(162)
plt.plot(delta, 1 - data1[2, :], label='Our Method', color='tomato', linestyle='--', marker='s')
plt.plot(delta, 1 - data1[3, :], label='BCUSUM', color='sandybrown', linestyle='-.', marker='s')
plt.legend(loc=2)
plt.title('n=200, $p_{n}=400$')
plt.subplot(163)
plt.plot(delta, 1 - data1[4, :], label='Our Method', color='gold', linestyle='--', marker='d')
plt.plot(delta, 1 - data1[5, :], label='BCUSUM', color='olive', linestyle='-.', marker='d')
plt.legend(loc=2)
plt.title('n=200, $p_{n}=600$')





plt.subplot(164)
plt.plot(delta, 1 - data1[6, :], label='Our Method', color='deepskyblue', linestyle='--', marker='o')
plt.plot(delta, 1 - data1[7, :], label='BCUSUM', color='royalblue', linestyle='-.', marker='o')
plt.legend(loc=2)
plt.title('n=500, $p_{n}=200$')
plt.subplot(165)
plt.plot(delta, 1 - data1[8, :], label='Our Method', color='tomato', linestyle='--', marker='s')
plt.plot(delta, 1 - data1[9, :], label='BCUSUM', color='sandybrown', linestyle='-.', marker='s')
plt.legend(loc=2)
plt.title('n=500, $p_{n}=400$')
plt.subplot(166)
plt.plot(delta, 1 - data1[10, :], label='Our Method', color='gold', linestyle='--', marker='d')
plt.plot(delta, 1 - data1[11, :], label='BCUSUM', color='olive', linestyle='-.', marker='d')
plt.legend(loc=2)
plt.title('n=500, $p_{n}=600$')
# plt.xlabel('Signal Size', fontsize=30)
# plt.ylabel('Power', fontsize=30)

fig.text(0.5, 0.02, 'Signal Size', ha='center',fontsize=18)
fig.text(0.08, 0.5, 'Power', va='center', rotation='vertical',fontsize=18)

plt.savefig("Power_Gaussian_sd.png")
plt.show()
