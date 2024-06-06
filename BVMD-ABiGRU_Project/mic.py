import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kendalltau
from scipy.stats import spearmanr
from minepy import MINE
# 准备时间序列数据
data1 = pd.read_csv('data/vmd.csv', encoding='gbk')
data2 = pd.read_csv('data/dwt.csv', encoding='gbk')
# P = data[data.columns[1]].values
# T = data[data.columns[2]].values
# SPEI = data[data.columns[6]].values
result=[]
print(data1.shape[1])
for i in range(0,data1.shape[1]):
    r= []
    for j in range(0,data2.shape[1]):
        mine = MINE()
        mine.compute_score(data1[data1.columns[i]].values, data2[data2.columns[j]].values)
        mic = mine.mic()
        r.append(mic)
        print("imf",i+1,"dwt",j,"MIC:", mic)
    result.append(r)
print(result)
# mine = MINE()
# mine.compute_score(K, Q)
# mic = mine.mic()
# print("MIC:", mic)
