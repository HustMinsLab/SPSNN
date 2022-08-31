import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
import pickle
import random
from pandas import DataFrame
import pandas as pd

# 该代码对样本求平均作为最后的样本，并扩展了AP
# 将sup分到train和test中
# #房间号
ROOM = 4

path1 = '..\\pickleData\\ss\\ave_subarea_data.pkl'
path2 = '..\\pickleData\\test\\ave_subarea_data.pkl'
path3 = '..\\pickleData\\sup\\ave_subarea_data.pkl'
with open(path1,'rb') as f:
    ave_subarea_data_train = pickle.load(f)
with open(path2,'rb') as f:
    ave_subarea_data_test = pickle.load(f)
with open(path3,'rb') as f:
    ave_subarea_data_sup = pickle.load(f)

# 每10s内的样本求平均作为最后的样本
room_train = ave_subarea_data_train[ROOM-1]
room_test = ave_subarea_data_test[ROOM-1]
room_sup = ave_subarea_data_sup[ROOM-1]

# 当x,y,localLabel一样时，说明时一个点在同一时间的采样
column_name_train = room_train.columns[1:].values.tolist()  # 数据框列名
column_name_test = room_test.columns[1:].values.tolist()  # 数据框列名
# 对于房间区域和走廊6
# 训练集样本
i = 0
rss_train = []
while i < len(room_train):
    x_label = room_train['xLabel'].iloc[i]
    y_label = room_train['yLabel'].iloc[i]
    localLabel = room_train['locaLabel'].iloc[i]
    roomLabel = room_train['RoomLabel'].iloc[i]
    devAdrLabel = room_train['DevAdrLabel'].iloc[i]
    rss_sample = []
    # 找到x,y,localLabel相同的RSS向量，求平均，作为最后样本
    rss = []
    for j in range(len(room_train)):
        if room_train['xLabel'].iloc[j] == x_label and room_train['yLabel'].iloc[j] == y_label and room_train['locaLabel'].iloc[i] == localLabel:
            rss.append(list(room_train.iloc[j,1:690]))
    temp_rss = np.array(rss)
    samples_amout = temp_rss.shape[0]
    for k in range(temp_rss.shape[1]):  # 遍历每一列（每个AP的RSS）
        sumt = np.sum(temp_rss[:,k])  # 该AP的所有RSS和
        if sumt < 0:  # 说明该AP收到了RSS
            rss_sample.append(sumt/np.count_nonzero(temp_rss[:,k]))  # 求平均作为该AP的栅格指纹
        else:
            rss_sample.append(0) # 该AP未收到RSS信号，置0
    rss_sample.append(x_label)
    rss_sample.append(y_label)
    rss_sample.append(localLabel)
    rss_sample.append(roomLabel)
    rss_sample.append(devAdrLabel)

    rss_train.append(rss_sample)
    i = i + samples_amout

room_train_new = DataFrame(data=rss_train,columns=column_name_train)

# 测试集样本
i = 0
rss_test = []
while i < len(room_test):
    x_label = room_test['xLabel'].iloc[i]
    y_label = room_test['yLabel'].iloc[i]
    localLabel = room_test['locaLabel'].iloc[i]
    roomLabel = room_test['RoomLabel'].iloc[i]
    devAdrLabel = room_test['DevAdrLabel'].iloc[i]
    rss_sample = []
    # 找到x,y,localLabel相同的RSS向量，求平均，作为最后样本
    rss = []
    for j in range(len(room_test)):
        if room_test['xLabel'].iloc[j] == x_label and room_test['yLabel'].iloc[j] == y_label and room_test['locaLabel'].iloc[i] == localLabel:
            rss.append(list(room_test.iloc[j,1:607]))
    temp_rss = np.array(rss)
    samples_amout = temp_rss.shape[0]
    for k in range(temp_rss.shape[1]):  # 遍历每一列（每个AP的RSS）
        sumt = np.sum(temp_rss[:,k])  # 该AP的所有RSS和
        if sumt < 0:  # 说明该AP收到了RSS
            rss_sample.append(sumt/np.count_nonzero(temp_rss[:,k]))  # 求平均作为该AP的栅格指纹
        else:
            rss_sample.append(0) # 该AP未收到RSS信号，置0
    rss_sample.append(x_label)
    rss_sample.append(y_label)
    rss_sample.append(localLabel)
    rss_sample.append(roomLabel)
    rss_sample.append(devAdrLabel)

    rss_test.append(rss_sample)
    i = i + samples_amout

room_test_new = DataFrame(data=rss_test,columns=column_name_test)

#----------------------------------------------扩展数据集AP情况---------------------------
column_name = room_sup.columns[1:].values.tolist()  # 数据框列名

new_data_train = []
c_name = room_train_new.columns.values.tolist()  # 数据框的列名
index = list(room_train_new.index)
for line in range(len(room_train_new)):
   temp=[]
   for col in column_name:
       if col in c_name:
           temp.append(room_train_new.loc[index[line], col])  # 取RSS向量
       else:
           temp.append(0)  # 如果别的AP不在train里面，则该AP的数据置0
   new_data_train.append(temp)

room_train_ext = DataFrame(data=new_data_train,columns=column_name)


new_data_test = []
c_name = room_test_new.columns.values.tolist()
index = list(room_test_new.index)
for line in range(len(room_test_new)):
   temp=[]
   for col in column_name:
       if col in c_name:
           temp.append(room_test_new.loc[index[line], col])
       else:
           temp.append(0)
   new_data_test.append(temp)

room_test_ext = DataFrame(data=new_data_test,columns=column_name)
del room_test
del room_train

path4 = '..\\pickleData\\ss\\ave\\train_room_'+ str(ROOM)+'.pkl'
path5 = '..\\pickleData\\test\\ave\\test_room_'+ str(ROOM)+'.pkl'
with open(path4,'wb') as f:
   pickle.dump(room_train_ext,f)
with open(path5,'wb') as f:
   pickle.dump(room_test_ext,f)
