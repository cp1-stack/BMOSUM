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
from scipy.stats import multivariate_t

matplotlib.rcParams['text.usetex'] = True
data = pd.read_csv('rmse_500_sd.csv')

plt.figure(figsize=(10, 6), dpi=300)
plt.tick_params(labelsize=14)
labels = ['Our Method', 'BCUSUM 200', 'Our 400', 'BCUSUM 400', 'Our 600', 'BCUSUM 600']
label1 = ['$p_{n}=200$', '$p_{n}=400$', '$p_{n}=600$']

bplt = plt.boxplot(data, notch=True, vert=True, patch_artist=True,labels=labels)
# colors = ['deepskyblue', 'royalblue', 'tomato', 'sandybrown', 'gold', 'olive']
colors = ['deepskyblue', 'deepskyblue', 'sandybrown', 'sandybrown', 'gold', 'gold']
for patch, color in zip(bplt['boxes'], colors):
    patch.set_facecolor(color)

# plt.xlabel('Alpha',fontsize=30)
plt.ylabel('RMSE', fontsize=18)
plt.legend(label1)
plt.savefig("box6.png")
plt.show()
