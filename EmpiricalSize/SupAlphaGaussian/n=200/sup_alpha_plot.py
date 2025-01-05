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

p=600
alpha = np.arange(0, 1, 0.1)
data = np.loadtxt(open("SupAlpha_p_{}_md.csv".format(p), "rb"), delimiter=",")

plt.figure(figsize=(10,8),dpi=300)
plt.tick_params(labelsize=20)

plt.plot(alpha, alpha, label='Alpha', color='deepskyblue', linestyle='-')
plt.plot(alpha, data[0,:], label='BMOSUM Empirical Approximation', color='tomato', linestyle='--',marker='o')
plt.plot(alpha, data[1,:], label='BCUSUM Empirical Approximation', color='palegreen', linestyle='-.',marker='^')
plt.plot(alpha, abs(alpha-data[0,:]), label='BMOSUM error-in-size', color='orange', linestyle='--',marker='o')
plt.plot(alpha, abs(alpha-data[1,:]), label='BCUSUM error-in-size', color='royalblue', linestyle='-.',marker='^')

# for a,b in zip(alpha,data[1,:]):
#     plt.text(a, b+0.001, '%.3f' % b, ha='center', va= 'bottom',fontsize=9)
plt.xlabel('Alpha',fontsize=30)
plt.ylabel('Empirical Approximation',fontsize=30)
plt.rcParams.update({'font.size': 18})
plt.legend()
plt.savefig("SupALpha_p_{}_md.png".format(p))
plt.show()


