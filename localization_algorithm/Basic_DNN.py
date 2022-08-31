# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 14:41:51 2019

@author: DELL PC
"""

import torch
import torch.nn.functional as F
import pickle
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
import random
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 无cuda,使用cpu
seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

ROOM = 4
data_set = 1  # 数据集序号

EPOCH = 1000           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 20
LR = 0.0001

Min_value = 100

AP_select = 0              #是否进行AP 筛选
threshold = 0             #AP筛选的阈值，仅在AP_select = 1时有效
disturb = 1  # 扰动数据
mean = 0
std = 250


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.mlp1 = torch.nn.Sequential(
                torch.nn.Linear(n_feature, n_hidden),
                torch.nn.ReLU(),
                )
        self.mlp2 = torch.nn.Sequential(
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.ReLU(),
                )
        self.mlp3 = torch.nn.Sequential(
                torch.nn.Linear(n_hidden, n_hidden),
#                torch.nn.Dropout(0.5),
                torch.nn.ReLU(),
                )
        self.out = torch.nn.Linear(n_hidden, n_output)
        
    def forward(self,x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.out(x)
        return x


path1 = '..\\pickleData\\raw_ratio\\average_data\\room_'+str(ROOM)+'_train.pkl'  # data set 1
path2 = '..\\pickleData\\raw_ratio\\average_data\\room_'+str(ROOM)+'_val.pkl'
path3 = '..\\pickleData\\raw_ratio\\average_data\\room_'+str(ROOM)+'_test.pkl'

model_path = '..\\model\\tmp_DNN\\disturbed\\Model For Room_'+str(ROOM)+'disturb'+str(std)+'.pkl'

with open(path1,'rb') as f:
    ratio_train = pickle.load(f)
with open(path2,'rb') as f:
    ratio_val = pickle.load(f)
with open(path3,'rb') as f:
    ratio_test = pickle.load(f)

if disturb==1:
    real = ratio_train.copy(deep=True)

    for i in range(ratio_train.shape[0]):
        add = np.random.normal(loc=mean, scale=std, size=2)
        ratio_train.loc[i,'xLabel'] = ratio_train.loc[i,'xLabel'] + add[0]
        ratio_train.loc[i,'yLabel'] = ratio_train.loc[i,'yLabel'] + add[1]

    disturb_path = '..\\pickleData\\raw_ratio\\average_data\\DNN\\room_' + str(ROOM) + '_disturb' + str(std) + '_train_disturbed.pkl'
    with open(disturb_path,'wb') as f:
        pickle.dump(ratio_train,f)
    
    diff = np.zeros( (ratio_train.shape[0],4), dtype=np.float )
    diff[:,0] = np.array(ratio_train.loc[:,'xLabel'])
    diff[:,1] = np.array(ratio_train.loc[:,'yLabel'])
    diff[:,2] = np.array(real.loc[:,'xLabel'])
    diff[:,3] = np.array(real.loc[:,'yLabel'])
    
    error = list(np.sqrt(np.square(diff[:,0]-diff[:,2])+np.square(diff[:,1]-diff[:,3])))




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


#训练集
if AP_select == 1:
    x,y,z = ratio_train.shape[0], Idx.count(True), 2
else:
    x,y,z = ratio_train.shape[0], ratio_train.shape[1]-6, 2
train_x = torch.zeros((x,y))
train_y = torch.zeros((x,z)) 
if disturb==1:
    train_z = torch.zeros((x,z))      #扰动时的truth
    for i in range(x):
        temp = np.array(ratio_train.iloc[i,1:ratio_train.shape[1]-5],dtype=float)
        if AP_select == 1:
            temp = temp[Idx]
            
        temp_y = np.array(ratio_train.iloc[i,ratio_train.shape[1]-5:ratio_train.shape[1]-3],dtype=float)
        temp_z = np.array(real.iloc[i,real.shape[1]-5:real.shape[1]-3],dtype=float)
    #    if np.std(temp) == 0:
    #        temp = temp - np.mean(temp)
    #    else:
    #        temp = (temp-np.mean(temp))/np.std(temp)
        
        train_x[i] = torch.from_numpy(temp)     
        train_y[i] = torch.from_numpy(temp_y) 
        train_z[i] = torch.from_numpy(temp_z) 
else:
    
    for i in range(x):
        temp = np.array(ratio_train.iloc[i,1:ratio_train.shape[1]-5],dtype=float)
        if AP_select == 1:
            temp = temp[Idx]
            
        temp_y = np.array(ratio_train.iloc[i,ratio_train.shape[1]-5:ratio_train.shape[1]-3],dtype=float)
    #    if np.std(temp) == 0:
    #        temp = temp - np.mean(temp)
    #    else:
    #        temp = (temp-np.mean(temp))/np.std(temp)
        
        train_x[i] = torch.from_numpy(temp)     
        train_y[i] = torch.from_numpy(temp_y) 

#验证集
if AP_select == 1:
    x,y,z = ratio_val.shape[0], Idx.count(True), 2
else:
    x,y,z = ratio_val.shape[0], ratio_val.shape[1]-6, 2
val_x = torch.zeros((x,y))
val_y = torch.zeros((x,z)) 
for i in range(x):
    temp = np.array(ratio_val.iloc[i,1:ratio_val.shape[1]-5],dtype=float)
    if AP_select == 1:
        temp = temp[Idx]
    temp_y = np.array(ratio_val.iloc[i,ratio_val.shape[1]-5:ratio_val.shape[1]-3],dtype=float)
#    if np.std(temp) == 0:
#        temp = temp - np.mean(temp)
#    else:
#        temp = (temp-np.mean(temp))/np.std(temp)
    
    val_x[i] = torch.from_numpy(temp)     
    val_y[i] = torch.from_numpy(temp_y)

#测试集
if AP_select == 1:
    x,y,z = ratio_test.shape[0], Idx.count(True), 2
else:
    x,y,z = ratio_test.shape[0], ratio_test.shape[1]-6, 2
test_x = torch.zeros((x,y))
test_y = torch.zeros((x,z)) 
for i in range(x):
    temp = np.array(ratio_test.iloc[i,1:ratio_test.shape[1]-5],dtype=float)
    if AP_select == 1:
        temp = temp[Idx]
    temp_y = np.array(ratio_test.iloc[i,ratio_test.shape[1]-5:ratio_test.shape[1]-3],dtype=float)
#    if np.std(temp) == 0:
#        temp = temp - np.mean(temp)
#    else:
#        temp = (temp-np.mean(temp))/np.std(temp)
    
    test_x[i] = torch.from_numpy(temp)     
    test_y[i] = torch.from_numpy(temp_y)

# val_x = val_x.cuda()
# val_y = val_y.cuda()
# test_x = test_x.cuda()
# test_y = test_y.cuda()

val_x = val_x.to(device)  # gpu转成cpu
val_y = val_y.to(device)
test_x = test_x.to(device)
test_y = test_y.to(device)

# net = Net(n_feature=y, n_hidden=300, n_output=2).cuda()
net = Net(n_feature=y, n_hidden=300, n_output=2).to(device)
print(net)

if disturb==1:
    train_data = Data.TensorDataset(train_x,train_y,train_z)
else:   
    train_data = Data.TensorDataset(train_x,train_y)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)    

#训练
optimizer = torch.optim.Adam(net.parameters(), lr=LR)   # optimize all parameters
loss_func = torch.nn.MSELoss()

if disturb==1:
    for epoch in range(EPOCH):
        net.train()
        for step, (b_x,b_y,b_z) in enumerate(train_loader):
            # b_x = b_x.cuda()
            # b_y = b_y.cuda()
            # b_z = b_z.cuda()

            b_x = b_x.to(device)  # [20,25,4],batchsize = 20
            b_y = b_y.to(device)  # [20,2
            b_z = b_z.to(device)  # [20,2
            output = net(b_x)
            loss = loss_func(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loc_err = torch.mean(torch.pow(torch.sum( torch.pow(output-b_z,2),1),0.5)/100)       
    #        print('Epoch: ', epoch, '| Step: ', step, 'loc_err:', loc_err)
        net.eval()
        test_output = net(val_x)
        loc_err1 = torch.mean(torch.pow(torch.sum( torch.pow(test_output-val_y,2),1),0.5)/100)
    #    print('Epoch: ', epoch,'| localization error for testdata:  ', loc_err)
        print('Epoch: ', epoch,'| train:  ', loc_err,'|  val: ',loc_err1)
        if float(loc_err1) <  Min_value:
            Min_value = float(loc_err1)
            torch.save(net, model_path)
else:
    for epoch in range(EPOCH):
        net.train()
        for step, (b_x,b_y) in enumerate(train_loader):
            # b_x = b_x.cuda()
            # b_y = b_y.cuda()
            b_x = b_x.to(device)  # [20,25,4],batchsize = 20
            b_y = b_y.to(device)  # [20,2
            output = net(b_x)
            loss = loss_func(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loc_err = torch.mean(torch.pow(torch.sum( torch.pow(output-b_y,2),1),0.5)/100)       
    #        print('Epoch: ', epoch, '| Step: ', step, 'loc_err:', loc_err)
        net.eval()
        test_output = net(val_x)
        loc_err1 = torch.mean(torch.pow(torch.sum( torch.pow(test_output-val_y,2),1),0.5)/100)
    #    print('Epoch: ', epoch,'| localization error for testdata:  ', loc_err)
        print('Epoch: ', epoch,'| train:  ', float(loc_err),'|  val: ',float(loc_err1))
        if float(loc_err1) <  Min_value:
            Min_value = float(loc_err1)
            torch.save(net, model_path)

final_net = torch.load(model_path)
with torch.no_grad():
    val_output = final_net(val_x)
loc_err = torch.mean(torch.pow(torch.sum( torch.pow(val_output-val_y,2),1),0.5)/100)
print('localization error for val data: %f'%loc_err)

with torch.no_grad():
    test_output = final_net(test_x)
loc_err1 = torch.mean(torch.pow(torch.sum( torch.pow(test_output-test_y,2),1),0.5)/100)
print('localization error for test data: %f'%loc_err1)

re_save = (torch.pow(torch.sum( torch.pow(test_output-test_y,2),1),0.5)/100).cpu().numpy()
save_path = '..\\pickleData\\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\DNN_'+str(std)+'_ROOM'+str(ROOM)+'.pkl'
with open(save_path,'wb') as f:
    pickle.dump(re_save,f)


print('--------------------room is  %d  --------------------------'%(ROOM))
print('--------------------data set is  %d  --------------------------'%(data_set))




