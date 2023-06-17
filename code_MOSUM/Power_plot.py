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

p=500

delta = np.arange(0, 2, 0.25)
data1 = np.loadtxt(open("Power_p_{}_m_250_id.csv".format(p), "rb"), delimiter=",")



plt.plot(delta, 1-data1[0,:], label='MOSUM', color='deepskyblue', linestyle='-.',marker='^')
plt.plot(delta, 1-data1[1,:], label='BMOSUM', color='tomato', linestyle='-.',marker='o')
plt.plot(delta, 1-data1[2,:], label='BCUSUM', color='mediumpurple', linestyle='-.',marker='s')



plt.xlabel('Signal Size')
plt.ylabel('Power')
plt.legend()
plt.savefig("Power_p_{}_id.png".format(p))
plt.show()


