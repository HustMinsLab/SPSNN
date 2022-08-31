import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
import pickle
import random
from pandas import DataFrame
import pandas as pd

# 读取csv，按房间分类，存成pkl
path_subarea_data_train = "../WiFi_data/NY/Mx5_NY/Mx5_SiteSurvey_NY.csv"
ave_subarea_data = pd.read_csv(path_subarea_data_train)

# path_ave_subarea_data_test = "../WiFi_data/NY/Mx5_NY/Mx5_Test_NY.csv"
# ave_subarea_data = pd.read_csv(path_ave_subarea_data_test)

# path_ave_subarea_data_sup = "../WiFi_data/NY/Mx5_NY/Mx5_Supplement_NY.csv"
# ave_subarea_data = pd.read_csv(path_ave_subarea_data_sup)

processing_data = []
# 按房间分类
count_1,count_2,count_3,count_4,count_5,count_6,count_7,count_8,count_9 = 0,0,0,0,0,0,0,0,0
for i in range(len(ave_subarea_data)):
    room_num = ave_subarea_data.loc[i,'RoomLabel']
    if room_num == 1:
        count_1 = count_1 + 1
    if room_num == 2:
        count_2 = count_2 + 1
    if room_num == 3:
        count_3 = count_3 + 1
    if room_num == 4:
        count_4 = count_4 + 1
    if room_num == 5:
        count_5 = count_5 + 1
    if room_num == 6:
        count_6 = count_6 + 1
    if room_num == 7:
        count_7 = count_7 + 1
    if room_num == 8:
        count_8 = count_8 + 1
    if room_num == 9:
        count_9 = count_9 + 1


processing_data.append(ave_subarea_data.iloc[0:count_1,:])
processing_data.append(ave_subarea_data.iloc[count_1:count_1 + count_2,:])
processing_data.append(ave_subarea_data.iloc[count_1 + count_2:count_1 + count_2 + count_3,:])
processing_data.append(ave_subarea_data.iloc[count_1 + count_2 + count_3:count_1 + count_2 + count_3 + count_4,:])
processing_data.append(ave_subarea_data.iloc[count_1 + count_2 + count_3 + count_4:count_1 + count_2 + count_3 + count_4 + count_5,:])
processing_data.append(ave_subarea_data.iloc[count_1 + count_2 + count_3 + count_4 + count_5:count_1 + count_2 + count_3 + count_4 + count_5 + count_6,:])
processing_data.append(ave_subarea_data.iloc[count_1 + count_2 + count_3 + count_4 + count_5 + count_6:count_1 + count_2 + count_3 + count_4 + count_5 + count_6 + count_7,:])
processing_data.append(ave_subarea_data.iloc[count_1 + count_2 + count_3 + count_4 + count_5 + count_6 + count_7:count_1 + count_2 + count_3 + count_4 + count_5 + count_6 + count_7 + count_8,:])
processing_data.append(ave_subarea_data.iloc[count_1 + count_2 + count_3 + count_4 + count_5 + count_6 + count_7 + count_8:count_1 + count_2 + count_3 + count_4 + count_5 + count_6 + count_7 + count_8 + count_9,:])




path1 = '..\\pickleData\\ss\\ave_subarea_data.pkl'
with open(path1,'wb') as f:
    pickle.dump(processing_data,f)

path2 = '..\\pickleData\\test\\ave_subarea_data.pkl'
with open(path2,'wb') as f:
    pickle.dump(processing_data,f)

path3 = '..\\pickleData\\sup\\ave_subarea_data.pkl'
with open(path3,'wb') as f:
    pickle.dump(processing_data,f)