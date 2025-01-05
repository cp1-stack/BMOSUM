import matplotlib.pyplot as plt
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
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns

data = pd.read_csv(r'acghData1.csv', sep=',', header=None)
X = data.values.T.astype(np.float32)
# y_ticks = ['1', '2', '3','4', '5','6','7','8','9','10','11','12','13','14',
# '15','16', '17', '18','19', '20','21','22','23','24','25','26','27','28','29',
# '30','31','32','33','34','35','36', '37', '38','39', '40','41','42','43']
plt.figure(figsize=(10, 5),dpi=300)
ax = sns.heatmap(X)
plt.axvline(317, color="yellow", ls='--')
plt.xlabel('Micro array index')
plt.ylabel('Patients')

plt.savefig("acghplot.png")
plt.show()