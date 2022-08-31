import pickle
import numpy as np
#import torch.utils.data as Data
import matplotlib.pyplot as plt
import random
from matplotlib.pyplot import MultipleLocator
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

# ROOM =1
data_set = 1
ERROR = 2
# std = 50
#调出预测结果
mean_err = np.zeros((6,4,7))
for room in range(4):
    for i in range(7):
        std = 50*i
        ROOM = room + 1

        path1 = '..\\pickleData\\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\NoTest_NN_ROOM'+str(ROOM)+'_disturb'+str(std)+'.pkl'
        path2 = '..\\pickleData\\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\DNN_'+str(std)+'_ROOM'+str(ROOM)+'.pkl'
        path3 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\SAE_DNN'+str(std)+'_ROOM'+str(ROOM)+'.pkl'
        path4 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\2DCNN'+str(std)+'_ROOM'+str(ROOM)+'.pkl'
        path5 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\CNNLoss_'+ str(std)+'_ROOM'+str(ROOM)+'.pkl'
        path6 = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\CNNEu_'+ str(std)+'_ROOM'+str(ROOM)+'.pkl'

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

        NN_err.append(NN)
        DNN_err.append(DNN)
        SAE_DNN_err.append(SAE_DNN)
        _2DCNN_err.append(_2DCNN)
        CNNLoss1_err.append(CNNLoss1)
        CNNEu_err.append(CNNEu)

        mean_err[0,ROOM-1,i] = np.mean(NN)
        mean_err[1,ROOM-1,i] = np.mean(DNN)
        mean_err[2,ROOM-1,i] = np.mean(SAE_DNN)
        mean_err[3,ROOM-1,i] = np.mean(_2DCNN)
        mean_err[4,ROOM-1,i] = np.mean(CNNEu)
        mean_err[5,ROOM-1,i] = np.mean(CNNLoss1)

all_err = np.zeros((6,7))
for i in range(len(mean_err)):
    all_err[i] = np.average(mean_err[i], axis=0)


a=['NN','DNN','SAE+DNN','2D CNN','S-CNN','SPCNN']
color_string = ['#F8C398', '#FED864', '#B6DCE7', '#C3D59F', '#E4B8B4', '#00AF4F']
fig = plt.figure(figsize=(10,8))
 	# 将画图窗口分成1行1列，选择第一块区域作子图
ax1 = fig.add_subplot(1, 1, 1)
 	# 设置标题
# ax1.set_title('Average Localization Error',fontsize=20)
 	# 设置横坐标名称
ax1.set_xlabel('标准差$\sigma$',fontsize=20)
 	# 设置纵坐标名称
ax1.set_ylabel('平均定位误差',fontsize=20)
 	# 画直线图
x = [0,50,100,150,200,250,300]
cl = ['grey','lightcoral','steelblue','m','darkgreen','darkorange']
# line_style = ['-','--','-.',':']
marker_line=['o','d','^','p','v','*']
for i in range(len(all_err)):  # NoVal_mean_err
    y = list(all_err[i])  # NoVal_mean_err[i]
    ax1.plot(x, y, c=color_string[i],marker= marker_line[i],label = a[i],linewidth=3,linestyle='-',markersize=15)

#plt.xlim(xmax=5, xmin=0)
 	# 显示
x_major_locator=MultipleLocator(50)
ax1.xaxis.set_major_locator(x_major_locator)
plt.grid()
plt.xticks(fontsize=11)
plt.yticks(fontsize=11)
# plt.axis('equal')
plt.legend(fontsize=13)
plt.show()

plt.savefig('..\\pickleData\\\localization_results\\figure\\ALE(3).png', bbox_inches='tight')