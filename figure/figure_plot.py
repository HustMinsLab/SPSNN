# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:08:25 2020

@author: DELL PC
"""

#import torch
#import torch.nn.functional as F
import pickle
import numpy as np
#import torch.utils.data as Data
import matplotlib.pyplot as plt
import random
from matplotlib.pyplot import MultipleLocator
# CNNEu 对应S-CNN
# CNNLoss 对应 SPSNN
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
ROOM =1
data_set = 1 # 数据集2被舍弃
ERROR = 2
std = 0
#调出预测结果
# path1 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\NoTest_NN_ROOM'+str(ROOM)+'.pkl'
# path2 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\DNN_ROOM'+str(ROOM)+'.pkl'
# path3 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\SAE_DNN_ROOM'+str(ROOM)+'.pkl'
# path4 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\2DCNN_ROOM'+str(ROOM)+'.pkl'
# path5 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNLoss_error'+str(ERROR)+'\\CNNLoss1_ROOM'+str(ROOM)+'.pkl'
# path6 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNEu_error'+str(ERROR)+'\\CNNEu_ROOM'+str(ROOM)+'.pkl'

path1 = '..\\pickleData\\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\NoTest_NN_ROOM'+str(ROOM)+'_disturb'+str(std)+'.pkl'
path2 = '..\\pickleData\\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\DNN_'+str(std)+'_ROOM'+str(ROOM)+'.pkl'
path3 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\SAE_DNN'+str(std)+'_ROOM'+str(ROOM)+'.pkl'
path4 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\2DCNN'+str(std)+'_ROOM'+str(ROOM)+'.pkl'
path5 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\CNNLoss_'+ str(std)+'_ROOM'+str(ROOM)+'.pkl'
path6 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\CNNEu_'+ str(std)+'_ROOM'+str(ROOM)+'.pkl'

# path1 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNLoss_error1\\CNNLoss1_ROOM'+str(ROOM)+'.pkl'
# path2 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNEu_error1\\CNNEu_ROOM'+str(ROOM)+'.pkl'
# path3 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNLoss_error2\\CNNLoss1_ROOM'+str(ROOM)+'.pkl'
# path4 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNEu_error2\\CNNEu_ROOM'+str(ROOM)+'.pkl'

with open(path1,'rb') as f:
    NN = pickle.load(f)
with open(path2,'rb') as f:
    DNN = pickle.load(f)
with open(path3,'rb') as f:
    SAE_DNN = pickle.load(f)
with open(path4,'rb') as f:
    _2DCNN = pickle.load(f)
with open(path5,'rb') as f:
    CNNLoss1 = pickle.load(f)
with open(path6,'rb') as f:
    CNNEu = pickle.load(f)


NN_err = []
DNN_err = []
SAE_DNN_err = []
_2DCNN_err = []
CNNLoss1_err = []
CNNEu_err = []

mean_err = np.zeros((6,4))
# mean_err = np.zeros((4,4))

NN_err.append(NN)
DNN_err.append(DNN)
SAE_DNN_err.append(SAE_DNN)
_2DCNN_err.append(_2DCNN)
CNNLoss1_err.append(CNNLoss1)
CNNEu_err.append(CNNEu)

mean_err[0,ROOM-1] = np.mean(NN)
mean_err[1,ROOM-1] = np.mean(DNN)
mean_err[2,ROOM-1] = np.mean(SAE_DNN)
mean_err[3,ROOM-1] = np.mean(_2DCNN)
mean_err[4,ROOM-1] = np.mean(CNNEu)
mean_err[5,ROOM-1] = np.mean(CNNLoss1)


ROOM =2
# path1 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\NoTest_NN_ROOM'+str(ROOM)+'.pkl'
# path2 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\DNN_ROOM'+str(ROOM)+'.pkl'
# path3 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\SAE_DNN_ROOM'+str(ROOM)+'.pkl'
# path4 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\2DCNN_ROOM'+str(ROOM)+'.pkl'
# path5 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNLoss_error'+str(ERROR)+'\\CNNLoss1_ROOM'+str(ROOM)+'.pkl'
# path6 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNEu_error'+str(ERROR)+'\\CNNEu_ROOM'+str(ROOM)+'.pkl'

path1 = '..\\pickleData\\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\NoTest_NN_ROOM'+str(ROOM)+'_disturb'+str(std)+'.pkl'
path2 = '..\\pickleData\\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\DNN_'+str(std)+'_ROOM'+str(ROOM)+'.pkl'
path3 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\SAE_DNN'+str(std)+'_ROOM'+str(ROOM)+'.pkl'
path4 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\2DCNN'+str(std)+'_ROOM'+str(ROOM)+'.pkl'
path5 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\CNNLoss_'+ str(std)+'_ROOM'+str(ROOM)+'.pkl'
path6 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\CNNEu_'+ str(std)+'_ROOM'+str(ROOM)+'.pkl'

# path1 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNLoss_error1\\CNNLoss1_ROOM'+str(ROOM)+'.pkl'
# path2 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNEu_error1\\CNNEu_ROOM'+str(ROOM)+'.pkl'
# path3 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNLoss_error2\\CNNLoss1_ROOM'+str(ROOM)+'.pkl'
# path4 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNEu_error2\\CNNEu_ROOM'+str(ROOM)+'.pkl'

with open(path1,'rb') as f:
    NN = pickle.load(f)
with open(path2,'rb') as f:
    DNN = pickle.load(f)
with open(path3,'rb') as f:
    SAE_DNN = pickle.load(f)
with open(path4,'rb') as f:
    _2DCNN = pickle.load(f)
with open(path5,'rb') as f:
    CNNLoss1 = pickle.load(f)
with open(path6,'rb') as f:
    CNNEu = pickle.load(f)

NN_err.append(NN)
DNN_err.append(DNN)
SAE_DNN_err.append(SAE_DNN)
_2DCNN_err.append(_2DCNN)
CNNLoss1_err.append(CNNLoss1)
CNNEu_err.append(CNNEu)

mean_err[0,ROOM-1] = np.mean(NN)
mean_err[1,ROOM-1] = np.mean(DNN)
mean_err[2,ROOM-1] = np.mean(SAE_DNN)
mean_err[3,ROOM-1] = np.mean(_2DCNN)
mean_err[4,ROOM-1] = np.mean(CNNEu)
mean_err[5,ROOM-1] = np.mean(CNNLoss1)
ROOM =3
# path1 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\NoTest_NN_ROOM'+str(ROOM)+'.pkl'
# path2 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\DNN_ROOM'+str(ROOM)+'.pkl'
# path3 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\SAE_DNN_ROOM'+str(ROOM)+'.pkl'
# path4 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\2DCNN_ROOM'+str(ROOM)+'.pkl'
# path5 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNLoss_error'+str(ERROR)+'\\CNNLoss1_ROOM'+str(ROOM)+'.pkl'
# path6 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNEu_error'+str(ERROR)+'\\CNNEu_ROOM'+str(ROOM)+'.pkl'

path1 = '..\\pickleData\\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\NoTest_NN_ROOM'+str(ROOM)+'_disturb'+str(std)+'.pkl'
path2 = '..\\pickleData\\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\DNN_'+str(std)+'_ROOM'+str(ROOM)+'.pkl'
path3 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\SAE_DNN'+str(std)+'_ROOM'+str(ROOM)+'.pkl'
path4 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\2DCNN'+str(std)+'_ROOM'+str(ROOM)+'.pkl'
path5 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\CNNLoss_'+ str(std)+'_ROOM'+str(ROOM)+'.pkl'
path6 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\CNNEu_'+ str(std)+'_ROOM'+str(ROOM)+'.pkl'

# path1 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNLoss_error1\\CNNLoss1_ROOM'+str(ROOM)+'.pkl'
# path2 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNEu_error1\\CNNEu_ROOM'+str(ROOM)+'.pkl'
# path3 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNLoss_error2\\CNNLoss1_ROOM'+str(ROOM)+'.pkl'
# path4 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNEu_error2\\CNNEu_ROOM'+str(ROOM)+'.pkl'

with open(path1,'rb') as f:
    NN = pickle.load(f)
with open(path2,'rb') as f:
    DNN = pickle.load(f)
with open(path3,'rb') as f:
    SAE_DNN = pickle.load(f)
with open(path4,'rb') as f:
    _2DCNN = pickle.load(f)
with open(path5,'rb') as f:
    CNNLoss1 = pickle.load(f)
with open(path6,'rb') as f:
    CNNEu = pickle.load(f)

NN_err.append(NN)
DNN_err.append(DNN)
SAE_DNN_err.append(SAE_DNN)
_2DCNN_err.append(_2DCNN)
CNNLoss1_err.append(CNNLoss1)
CNNEu_err.append(CNNEu)

mean_err[0,ROOM-1] = np.mean(NN)
mean_err[1,ROOM-1] = np.mean(DNN)
mean_err[2,ROOM-1] = np.mean(SAE_DNN)
mean_err[3,ROOM-1] = np.mean(_2DCNN)
mean_err[4,ROOM-1] = np.mean(CNNEu)
mean_err[5,ROOM-1] = np.mean(CNNLoss1)
ROOM =4
# path1 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\NoTest_NN_ROOM'+str(ROOM)+'.pkl'
# path2 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\DNN_ROOM'+str(ROOM)+'.pkl'
# path3 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\SAE_DNN_ROOM'+str(ROOM)+'.pkl'
# path4 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\2DCNN_ROOM'+str(ROOM)+'.pkl'
# path5 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNLoss_error'+str(ERROR)+'\\CNNLoss1_ROOM'+str(ROOM)+'.pkl'
# path6 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNEu_error'+str(ERROR)+'\\CNNEu_ROOM'+str(ROOM)+'.pkl'

path1 = '..\\pickleData\\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\NoTest_NN_ROOM'+str(ROOM)+'_disturb'+str(std)+'.pkl'
path2 = '..\\pickleData\\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\DNN_'+str(std)+'_ROOM'+str(ROOM)+'.pkl'
path3 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\SAE_DNN'+str(std)+'_ROOM'+str(ROOM)+'.pkl'
path4 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\2DCNN'+str(std)+'_ROOM'+str(ROOM)+'.pkl'
path5 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\CNNLoss_'+ str(std)+'_ROOM'+str(ROOM)+'.pkl'
path6 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\CNNEu_'+ str(std)+'_ROOM'+str(ROOM)+'.pkl'


# path1 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNLoss_error1\\CNNLoss1_ROOM'+str(ROOM)+'.pkl'
# path2 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNEu_error1\\CNNEu_ROOM'+str(ROOM)+'.pkl'
# path3 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNLoss_error2\\CNNLoss1_ROOM'+str(ROOM)+'.pkl'
# path4 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\CNNEu_error2\\CNNEu_ROOM'+str(ROOM)+'.pkl'

with open(path1,'rb') as f:
    NN = pickle.load(f)
with open(path2,'rb') as f:
    DNN = pickle.load(f)
with open(path3,'rb') as f:
    SAE_DNN = pickle.load(f)
with open(path4,'rb') as f:
    _2DCNN = pickle.load(f)
with open(path5,'rb') as f:
    CNNLoss1 = pickle.load(f)
with open(path6,'rb') as f:
    CNNEu = pickle.load(f)

NN_err.append(NN)
DNN_err.append(DNN)
SAE_DNN_err.append(SAE_DNN)
_2DCNN_err.append(_2DCNN)
CNNLoss1_err.append(CNNLoss1)
CNNEu_err.append(CNNEu)

mean_err[0,ROOM-1] = np.mean(NN)
mean_err[1,ROOM-1] = np.mean(DNN)
mean_err[2,ROOM-1] = np.mean(SAE_DNN)
mean_err[3,ROOM-1] = np.mean(_2DCNN)
mean_err[4,ROOM-1] = np.mean(CNNEu)
mean_err[5,ROOM-1] = np.mean(CNNLoss1)

All_room = []
All_room.append(np.concatenate((NN_err[0] ,NN_err[1],NN_err[2],NN_err[3])))
All_room.append(np.concatenate((DNN_err[0] ,DNN_err[1],DNN_err[2],DNN_err[3])))
All_room.append(np.concatenate((SAE_DNN_err[0] ,SAE_DNN_err[1],SAE_DNN_err[2],SAE_DNN_err[3])))
All_room.append(np.concatenate((_2DCNN_err[0] ,_2DCNN_err[1],_2DCNN_err[2],_2DCNN_err[3])))
All_room.append(np.concatenate((CNNLoss1_err[0] ,CNNLoss1_err[1],CNNLoss1_err[2],CNNLoss1_err[3])))
All_room.append(np.concatenate((CNNEu_err[0] ,CNNEu_err[1],CNNEu_err[2],CNNEu_err[3])))


#画图
dataSets=[All_room[0],All_room[1],All_room[2],All_room[3],All_room[5],All_room[4]]
# dataSets=[All_room[0],All_room[1],All_room[2],All_room[3]]

temp = []
for set in dataSets:
    temp2 = []
    for item in set:
        if item!='':
            temp2.append(float(item))
    temp2.sort()
    temp.append(temp2)
dataSets = temp


# CDF曲线
a=['NN','DNN','SAE+DNN','2D CNN','S-CNN','SPCNN']
color_string = ['#F8C398', '#FED864', '#B6DCE7', '#C3D59F', '#E4B8B4', '#00AF4F']
marker_line=['o','d','^','p','v','*']
plt.figure(figsize=(10,8))
for n, set in enumerate(dataSets):
    label=0
    plotDataset = [[],[]]
    count = len(set)
    for i in range(count):
        plotDataset[0].append(float(set[i]))
        plotDataset[1].append((i+1)/count)
        if ((plotDataset[1][i]>=0.9)&(label==0)):
            label=1
            print (plotDataset[0][i])
            
    #print(plotDataset)
    if(n==0):
        plt.plot(plotDataset[0], plotDataset[1], linestyle='-', color=color_string[0],linewidth=3,label = a[n])
    if(n==1):
        plt.plot(plotDataset[0], plotDataset[1], linestyle='-', color=color_string[1],linewidth=3,label = a[n])
    if(n==2):
        plt.plot(plotDataset[0], plotDataset[1], linestyle='-', color=color_string[2],linewidth=3,label = a[n])
    if(n==3):
        plt.plot(plotDataset[0], plotDataset[1], linestyle='-', color=color_string[3],linewidth=3,label = a[n])
    if(n==4):
        plt.plot(plotDataset[0], plotDataset[1], linestyle='-', color=color_string[4],linewidth=3,label = a[n])
    if(n==5):
        plt.plot(plotDataset[0], plotDataset[1], linestyle='-', color=color_string[5],linewidth=3,label = a[n])

        
    plt.xlabel('Distance Estimation Error (m)',fontsize=15)
    plt.ylabel('Cumulative Distribution Function',fontsize=15)
    # plt.title('Cumulative Distribution Function',fontsize=20)
    plt.xlim((-0.01, 3))
    plt.ylim((-0.01, 1.02))
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right',fontsize=13)
    i=i+1

plt.savefig('..\\pickleData\\\localization_results\\figure\\CDF(1).png', bbox_inches='tight')
# plt.savefig('..\\pickleData\\\localization_results\\figure\\CDF(2).png', bbox_inches='tight')

#房间级别的曲线
# 画ALE
fig = plt.figure(figsize=(10,8))
 	# 将画图窗口分成1行1列，选择第一块区域作子图
ax1 = fig.add_subplot(1, 1, 1)
 	# 设置标题
# ax1.set_title('Average Localization Error',fontsize=20)
 	# 设置横坐标名称
ax1.set_xlabel('房间序号',fontsize=20)
 	# 设置纵坐标名称
ax1.set_ylabel('平均定位误差',fontsize=20)
 	# 画直线图
x = [1,2,3,4]
cl = ['grey','lightcoral','steelblue','m','darkgreen','darkorange']
marker_line=['o','d','^','p','v','*']
for i in range(len(mean_err)):  # NoVal_mean_err
    y = list(mean_err[i])  # NoVal_mean_err[i]
    # ax1.plot(x, y, c=color_string[i],marker= '.',label = a[i],linewidth=3)
    ax1.plot(x, y, c=color_string[i], marker=marker_line[i], label=a[i], linewidth=3, linestyle='-', markersize=15)

# plt.xlim(xmax=3, xmin=0)
 	# 显示
x_major_locator=MultipleLocator(1)
ax1.xaxis.set_major_locator(x_major_locator)
plt.grid()
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
# plt.axis('equal')
plt.legend(fontsize=13)
plt.savefig('..\\pickleData\\\localization_results\\figure\\ALE(1).png', bbox_inches='tight')
# plt.savefig('..\\pickleData\\\localization_results\\figure\\ALE(2).png', bbox_inches='tight')
plt.show()
