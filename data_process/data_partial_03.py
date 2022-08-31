import pickle
import random
import pandas as pd

# 房间号
ROOM = 4
data_set = 1
ratio = [0.7,0.2,0.1]

path1 = '..\\pickleData\\ss\\ave\\train_room_'+ str(ROOM)+'.pkl'
path2 = '..\\pickleData\\test\\ave\\test_room_'+ str(ROOM)+'.pkl'

with open(path1,'rb') as f:
    room_train = pickle.load(f)
with open(path2,'rb') as f:
    room_test = pickle.load(f)

#1.数据索引分类
index_train,index_val,index_test = [], [], []

#1.1 train数据集
index = list(room_train.index)
random.shuffle(index)
train_num, val_num = int(ratio[0]*len(index)),int(ratio[1]*len(index))
test_num = len(index) - train_num - val_num
index_train.append(index[0:train_num])
index_val.append(index[train_num:train_num+val_num])
index_test.append(index[train_num+val_num:])
#
# 1.2 test数据集
index = list(room_test.index)
random.shuffle(index)
train_num, val_num = int(ratio[0]*len(index)),int(ratio[1]*len(index))
test_num = len(index) - train_num - val_num
index_train.append(index[0:train_num])
index_val.append(index[train_num:train_num+val_num])
index_test.append(index[train_num+val_num:])

# #ratio_train数据集
ratio_train = pd.concat([room_train.loc[index_train[0]],room_test.loc[index_train[1]]], ignore_index=True)
ratio_val = pd.concat([room_train.loc[index_val[0]],room_test.loc[index_val[1]]], ignore_index=True)
ratio_test = pd.concat([room_train.loc[index_test[0]],room_test.loc[index_test[1]]], ignore_index=True)

# 按比例划分的数据，把训练集，测试集，扩展数据的0.7作为训练集，0.2作为验证集，0.1作为测试集
path3 = '..\\pickleData\\raw_ratio\\average_data\\room_'+str(ROOM)+'_train.pkl'
path4 = '..\\pickleData\\raw_ratio\\average_data\\room_'+str(ROOM)+'_val.pkl'
path5 = '..\\pickleData\\raw_ratio\\average_data\\room_'+str(ROOM)+'_test.pkl'

with open(path3,'wb') as f:
   pickle.dump(ratio_train,f)
with open(path4,'wb') as f:
   pickle.dump(ratio_val,f)
with open(path5,'wb') as f:
   pickle.dump(ratio_test,f)

