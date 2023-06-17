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

data = pd.read_csv('rmse_500_sd_t.csv')

plt.figure(figsize=(10, 6), dpi=300)
plt.tick_params(labelsize=10)
labels = ['BMOSUM_200', 'BCUSUM_200', 'BMOSUM_400', 'BCUSUM_400', 'BMOSUM_600', 'BCUSUM_600']

bplt = plt.boxplot(data, notch=True, vert=True, patch_artist=True,labels=labels)
# colors = ['deepskyblue', 'royalblue', 'tomato', 'sandybrown', 'gold', 'olive']
colors = ['deepskyblue', 'deepskyblue', 'sandybrown', 'sandybrown', 'gold', 'gold']
for patch, color in zip(bplt['boxes'], colors):
    patch.set_facecolor(color)

# plt.xlabel('Alpha',fontsize=30)
plt.ylabel('RMSE', fontsize=30)
# plt.rcParams.update({'font.size': 18})
# plt.legend()
plt.savefig("box6_t.png")
plt.show()
