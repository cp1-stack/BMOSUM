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

p=200

delta = np.arange(0, 2, 0.25)
data1 = np.loadtxt(open("Power_p_{}_m_250_id.csv".format(p), "rb"), delimiter=",")

plt.figure(figsize=(10,8),dpi=300)
plt.tick_params(labelsize=20)

plt.plot(delta, 1-data1[0,:], label='BMOSUM', color='tomato', linestyle='--',marker='o')
plt.plot(delta, 1-data1[1,:], label='BCUSUM', color='mediumpurple', linestyle='-.',marker='s')



plt.xlabel('Signal Size',fontsize=30)
plt.ylabel('Power',fontsize=30)
plt.rcParams.update({'font.size': 18})
plt.legend()
plt.savefig("Power_p_{}_id.png".format(p))
plt.show()


