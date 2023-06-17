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
p=300


data1 = np.loadtxt(open("Power_p_300_m_50_md.csv", "rb"), delimiter=",")
data2 = np.loadtxt(open("Power_p_300_m_100_md.csv", "rb"), delimiter=",")
data3 = np.loadtxt(open("Power_p_300_m_250_md.csv", "rb"), delimiter=",")


plt.plot(np.arange(0, 2, 0.25), data1[1,:], label='Location=0.1', color='deepskyblue', linestyle='-.',marker='^')
plt.plot(np.arange(0, 2, 0.25), data2[1,:], label='Location=0.2', color='tomato', linestyle='--',marker='o')
plt.plot(np.arange(0, 2, 0.25), data3[1,:], label='Location=0.5', color='mediumpurple', linestyle='--',marker='s')


plt.xlabel('Signal Size')
plt.ylabel('Power')
plt.legend()
plt.savefig("Power_p_{}_md.png".format(p))
plt.show()


