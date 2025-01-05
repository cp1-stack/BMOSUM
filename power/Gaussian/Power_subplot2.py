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
plt.figure(figsize=(10, 6), dpi=300)


# plt.plot(delta, 1 - data1[6, :], label='Our Method', color='deepskyblue', linestyle='--', marker='o',linewidth=2)
# plt.plot(delta, 1 - data1[7, :], label='BCUSUM', color='royalblue', linestyle='-.', marker='o',linewidth=2)


plt.plot(delta, 1 - data1[8, :], label='Our Method', color='tomato', linestyle='--', marker='s',linewidth=2)
plt.plot(delta, 1 - data1[9, :], label='BCUSUM', color='sandybrown', linestyle='-.', marker='s',linewidth=2)


# plt.plot(delta, 1 - data1[10, :], label='Our Method', color='seagreen', linestyle='--', marker='d',linewidth=2)
# plt.plot(delta, 1 - data1[11, :], label='BCUSUM', color='olive', linestyle='-.', marker='d',linewidth=2)

plt.legend(prop = {'size':20},loc=2)

plt.xlabel('Signal Size', fontsize=30)
plt.ylabel('Power', fontsize=30)



plt.savefig("Power_Gaussian_sd5.png")
plt.show()
