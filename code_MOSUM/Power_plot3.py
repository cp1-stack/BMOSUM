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
p=1000

delta = np.arange(0, 2, 0.25)
data1 = np.loadtxt(open("Power_p_{}_m_250_sd.csv".format(p), "rb"), delimiter=",")


plt.plot(delta, 1-data1[0,:], label='MOSUM', color='deepskyblue', linestyle='-.',marker='^')
plt.plot(delta, 1-data1[1,:], label='BMOSUM', color='tomato', linestyle='-.',marker='o')
plt.plot(delta, 1-data1[2,:], label='BCUSUM', color='mediumpurple', linestyle='-.',marker='s')
plt.plot(delta, 1-data1[3,:], label='CUSUM', color='y', linestyle='-.',marker='>')

plt.xlabel('Signal Size')
plt.ylabel('Power')
plt.legend()
plt.savefig("Power_p_{}_sd.png".format(p))
plt.show()


