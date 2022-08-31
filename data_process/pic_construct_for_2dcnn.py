import torch
import torch.nn.functional as F
import pickle
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
import random
import torch.nn as nn
import math
from sklearn import preprocessing

# 2DCNN输入图像构建
seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

ROOM = 4
data_set = 1
std = 250


#-----------------------------------------------------------------------------------------------------------
#---------------------------------------------------图片构造-------------------------------------------------
# AP_select = 1              #是否进行AP 筛选
threshold = 0             #AP筛选的阈值，仅在AP_select = 1时有效

path1 = '..\\pickleData\\raw_ratio\\average_data\\disturbed_data\\room_'+str(ROOM)+'_disturb'+str(std)+'_train_disturbed.pkl'
path2 = '..\\pickleData\\raw_ratio\\average_data\\room_'+str(ROOM)+'_val.pkl'
path3 = '..\\pickleData\\raw_ratio\\average_data\\room_'+str(ROOM)+'_test.pkl'

with open(path1,'rb') as f:
    ratio_train = pickle.load(f)
with open(path2,'rb') as f:
    ratio_val = pickle.load(f)
with open(path3,'rb') as f:
    ratio_test = pickle.load(f)


coverage_count = [0]*761
for i in range(len(ratio_train)):
    test_rss = list(ratio_train.iloc[i,1:762])
    for k in range(len(test_rss)):
        if test_rss[k] != 0:
            coverage_count[k] += 1

receive_ratio = list(np.array(coverage_count)/len(ratio_train))

Idx = []
for i in range(len(receive_ratio)):
    if receive_ratio[i] > threshold:
        Idx.append(True)
    else:
        Idx.append(False)

#构造图片
#训练集
picture_data_train_06 = [[None,None] for x in range(ratio_train.shape[0])]
for i in range(ratio_train.shape[0]):
    temp = np.array(list(ratio_train.iloc[i,1:762]))
    RSS_Raw = []
    for r in range(len(temp)):
        if Idx[r] == False:
            continue
        else:
            if temp[r] == 0:
                RSS_Raw.append(-110)
            else:
                RSS_Raw.append(temp[r])
    p = math.ceil(pow(len(RSS_Raw),0.5))
    padding = np.zeros(p**2 -len(RSS_Raw) )
    RSS_Raw  = np.hstack((RSS_Raw ,padding))
    RSS_Norm = preprocessing.robust_scale(np.array(RSS_Raw))
    pic = RSS_Norm.reshape((p,p))
    picture_data_train_06[i][0],picture_data_train_06[i][1] = pic, [ratio_train.loc[i,'xLabel'], ratio_train.loc[i,'yLabel']]

#验证集
picture_data_val_06 = [[None,None] for x in range(ratio_val.shape[0])]
for i in range(ratio_val.shape[0]):
    temp = np.array(list(ratio_val.iloc[i,1:762]))
    RSS_Raw = []
    for r in range(len(temp)):
        if Idx[r] == False:
            continue
        else:
            if temp[r] == 0:
                RSS_Raw.append(-110)
            else:
                RSS_Raw.append(temp[r])
    p = math.ceil(pow(len(RSS_Raw),0.5))
    padding = np.zeros(p**2 -len(RSS_Raw) )
    RSS_Raw  = np.hstack((RSS_Raw ,padding))
    RSS_Norm = preprocessing.robust_scale(np.array(RSS_Raw))
    pic = RSS_Norm.reshape((p,p))
    picture_data_val_06[i][0],picture_data_val_06[i][1] = pic, [ratio_val.loc[i,'xLabel'], ratio_val.loc[i,'yLabel']]
#测试集
picture_data_test_06 = [[None,None] for x in range(ratio_test.shape[0])]
for i in range(ratio_test.shape[0]):
    temp = np.array(list(ratio_test.iloc[i,1:762]))
    RSS_Raw = []
    for r in range(len(temp)):
        if Idx[r] == False:
            continue
        else:
            if temp[r] == 0:
                RSS_Raw.append(-110)
            else:
                RSS_Raw.append(temp[r])
    p = math.ceil(pow(len(RSS_Raw),0.5))
    padding = np.zeros(p**2 -len(RSS_Raw) )
    RSS_Raw  = np.hstack((RSS_Raw ,padding))
    RSS_Norm = preprocessing.robust_scale(np.array(RSS_Raw))
    pic = RSS_Norm.reshape((p,p))
    picture_data_test_06[i][0],picture_data_test_06[i][1] = pic, [ratio_test.loc[i,'xLabel'], ratio_test.loc[i,'yLabel']]

path11 = '..\\pickleData\\raw_ratio\\pics\\2D_CNN\\train_room_'+ str(ROOM)+'_disturb'+str(std)+'.pkl'
path13 = '..\\pickleData\\raw_ratio\\pics\\2D_CNN\\val_room_'+ str(ROOM)+'.pkl'
path15 = '..\\pickleData\\raw_ratio\\pics\\2D_CNN\\test_room_'+ str(ROOM)+'.pkl'

with open(path11,'wb') as f:
    pickle.dump(picture_data_train_06,f)

with open(path13,'wb') as f:
    pickle.dump(picture_data_val_06,f)

with open(path15,'wb') as f:
    pickle.dump(picture_data_test_06,f)


print('--------------------room is  %d  --------------------------'%(ROOM))
print('--------------------data set is  %d  --------------------------'%(data_set))
#-----------------------------------------------------------------------------------------------------------
#---------------------------------------------------图片构造 end-------------------------------------------------