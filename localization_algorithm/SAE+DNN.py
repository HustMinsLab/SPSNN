# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:50:36 2020

@author: DELL PC
"""


import torch
import pickle
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
import random
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 无cuda,使用cpu

seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

ROOM = 4
data_set = 1
std = 250

EPOCH_SAE1 = 600
EPOCH_SAE2 = 600
EPOCH_SAE3 = 600
EPOCH = 500       # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 32
LR_SAE1 = 0.001
LR_SAE2 = 0.001
LR_SAE3 = 0.001
LR = 0.01

Min_value = 1000

AP_select = 1              #是否进行AP 筛选
threshold = 0             #AP筛选的阈值，仅在AP_select = 1时有效

para_dict = {}

path1 = '..\\pickleData\\raw_ratio\\average_data\\disturbed_data\\room_'+str(ROOM)+'_disturb'+str(std)+'_train_disturbed.pkl'  # 扰动数据
path2 = '..\\pickleData\\raw_ratio\\average_data\\room_'+str(ROOM)+'_val.pkl'
path3 = '..\\pickleData\\raw_ratio\\average_data\\room_'+str(ROOM)+'_test.pkl'

model_path_SAE_1 = '..\\model\\SAE+DNN\\data_set_'+str(data_set)+'\\disturbed\\SAE_1_'+str(std)+'Model For  Room_'+str(ROOM)+'.pkl'
model_path_SAE_2 = '..\\model\\SAE+DNN\\data_set_'+str(data_set)+'\\disturbed\\SAE_2_'+str(std)+'Model For  Room_'+str(ROOM)+'.pkl'
model_path_SAE_3 = '..\\model\\SAE+DNN\\data_set_'+str(data_set)+'\\disturbed\\SAE_3_'+str(std)+'Model For  Room_'+str(ROOM)+'.pkl'
model_path = '..\\model\\SAE+DNN\\data_set_'+str(data_set)+'\\disturbed\\SAE_DNN'+str(std)+'_Model For  Room_'+str(ROOM)+'.pkl'

with open(path1,'rb') as f:
    ratio_train = pickle.load(f)
with open(path2,'rb') as f:
    ratio_val = pickle.load(f)
with open(path3,'rb') as f:
    ratio_test = pickle.load(f)

coverage_count = [0]*761
for i in range(len(ratio_train)):
    test_rss = list(ratio_train.iloc[i,:761])
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

    
for i in range(x):
    temp = np.array(ratio_train.iloc[i,:ratio_train.shape[1]-5],dtype=float)
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
    temp = np.array(ratio_val.iloc[i,:ratio_val.shape[1]-5],dtype=float)
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
    temp = np.array(ratio_test.iloc[i,:ratio_test.shape[1]-5],dtype=float)
    if AP_select == 1:
        temp = temp[Idx]
    temp_y = np.array(ratio_test.iloc[i,ratio_test.shape[1]-5:ratio_test.shape[1]-3],dtype=float)
#    if np.std(temp) == 0:
#        temp = temp - np.mean(temp)
#    else:
#        temp = (temp-np.mean(temp))/np.std(temp)
    
    test_x[i] = torch.from_numpy(temp)     
    test_y[i] = torch.from_numpy(temp_y)


   


class AE(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(AE, self).__init__()
        self.mlp1 = torch.nn.Sequential(
                torch.nn.Linear(n_feature, n_hidden),
                torch.nn.ReLU(),
                )   
        self.out = torch.nn.Linear(n_hidden, n_output)
    def forward(self,x):
        x = self.mlp1(x)
        x1 = self.out(x)
        return x1,x

class AE1(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(AE1, self).__init__()
        self.mlp1 = torch.nn.Sequential(
                torch.nn.Linear(n_feature, n_hidden),
                torch.nn.ReLU(),
                )   
        self.out = torch.nn.Linear(n_hidden, n_output)
    def forward(self,x):
        x = self.mlp1(x)
        x1 = self.out(x)
        return x1,x
class AE2(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(AE2, self).__init__()
        self.mlp1 = torch.nn.Sequential(
                torch.nn.Linear(n_feature, n_hidden),
                torch.nn.ReLU(),
                )   
        self.out = torch.nn.Linear(n_hidden, n_output)
    def forward(self,x):
        x = self.mlp1(x)
        x1 = self.out(x)
        return x1,x

class SAE_DNN(torch.nn.Module):
    def __init__(self, n_feature,n_AE1, n_AE2, n_AE3,n_D,n_output,para_dict):
        super(SAE_DNN, self).__init__()
        
        
        self.sae1 = torch.nn.Sequential(
                torch.nn.Linear(n_feature, n_AE1),
                torch.nn.ReLU(),
                ) 
        self.sae2 = torch.nn.Sequential(
                torch.nn.Linear(n_AE1, n_AE2),
                torch.nn.ReLU(),
                ) 
        self.sae3 = torch.nn.Sequential(
                torch.nn.Linear(n_AE2, n_AE3),
                torch.nn.ReLU(),
                ) 
        self.mlp1 = torch.nn.Sequential(
                torch.nn.Linear(n_AE3, n_D),
                torch.nn.ReLU(),
                ) 
        self.mlp2 = torch.nn.Sequential(
                torch.nn.Linear(n_D, n_D),
                torch.nn.ReLU(),
                ) 
        self.out = torch.nn.Linear(n_D, n_output) 
        
        
        #SAE参数初始化
        self.sae1[0].weight,self.sae1[0].bias  = nn.Parameter(para_dict['AE1_weight']),nn.Parameter(para_dict['AE1_bias'])
        self.sae2[0].weight,self.sae2[0].bias  = nn.Parameter(para_dict['AE2_weight']),nn.Parameter(para_dict['AE2_bias'])
        self.sae3[0].weight,self.sae3[0].bias  =nn.Parameter( para_dict['AE3_weight']),nn.Parameter(para_dict['AE3_bias'])
        
        
    def forward(self, x):
        x = self.sae1(x)
        x = self.sae2(x)
        x = self.sae3(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.out(x)
        return x




N_INPUT1 = Idx.count(True)
N_HIDDEN_1 = 256
N_OUTPUT_1 = N_INPUT1

N_INPUT2 = N_HIDDEN_1
N_HIDDEN_2 = 128
N_OUTPUT_2 = N_INPUT2

N_INPUT3 = N_HIDDEN_2
N_HIDDEN_3 = 64
N_OUTPUT_3 = N_INPUT3


train_data = Data.TensorDataset(train_x,train_y)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True) 

SAE_1 = AE(N_INPUT1,N_HIDDEN_1,N_OUTPUT_1)
optimizer = torch.optim.Adam(SAE_1.parameters(), lr=LR_SAE1)   # optimize all cnn parameters
loss_func = nn.MSELoss()


for epoch in range(EPOCH_SAE1):
    SAE_1.train()
    for step, (b_x,b_y) in enumerate(train_loader):
        output,_ = SAE_1(b_x)
        loss = loss_func(output,b_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
             
    SAE_1.eval()
    train_output,_ = SAE_1(train_x)
    loc_err1 = loss_func(train_output,train_x)
    print('SAE_1 Training LOSS %f'%loc_err1)
    if float(loc_err1) <  Min_value:
        Min_value = float(loc_err1)
        torch.save(SAE_1, model_path_SAE_1)
    
final_net1 = torch.load(model_path_SAE_1)
_,h1 = final_net1(train_x)
para_dict['AE1_weight']=final_net1.state_dict()['mlp1.0.weight']
para_dict['AE1_bias']=final_net1.state_dict()['mlp1.0.bias']

del train_data
del train_loader


Min_value = 1000
train_data = Data.TensorDataset(h1.data,train_y)    #train_y不重要
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)  


SAE_2 = AE1(N_INPUT2,N_HIDDEN_2,N_OUTPUT_2)
optimizer = torch.optim.Adam(SAE_2.parameters(), lr=LR_SAE2)   # optimize all cnn parameters
loss_func = nn.MSELoss()


for epoch in range(EPOCH_SAE2):

    SAE_2.train()
    for step, (b_x,b_y) in enumerate(train_loader):
        output,_ = SAE_2(b_x)
        loss = loss_func(output,b_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
             
    SAE_2.eval()
    train_output,_ = SAE_2(h1)
    loc_err1 = loss_func(train_output,h1)
    print('SAE_2 Training LOSS %f'%loc_err1)
    if float(loc_err1) <  Min_value:
        Min_value = float(loc_err1)
        torch.save(SAE_2, model_path_SAE_2)
    
final_net2 = torch.load(model_path_SAE_2)
_,h2 = final_net2(h1)
para_dict['AE2_weight']=final_net2.state_dict()['mlp1.0.weight']
para_dict['AE2_bias']=final_net2.state_dict()['mlp1.0.bias']

del train_data
del train_loader

Min_value = 1000
train_data = Data.TensorDataset(h2.data,train_y)    #train_ybu不重要
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)  


SAE_3 = AE2(N_INPUT3,N_HIDDEN_3,N_OUTPUT_3)
optimizer = torch.optim.Adam(SAE_3.parameters(), lr=LR_SAE3 )   # optimize all cnn parameters
loss_func = nn.MSELoss()


for epoch in range(EPOCH_SAE3):
    SAE_3.train()
    for step, (b_x,b_y) in enumerate(train_loader):
        output,_ = SAE_3(b_x)
        loss = loss_func(output,b_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
             
    SAE_3.eval()
    train_output,_ = SAE_3(h2)
    loc_err1 = loss_func(train_output,h2)
    print('SAE_3 Training LOSS %f'%loc_err1)
    if float(loc_err1) <  Min_value:
        Min_value = float(loc_err1)
        torch.save(SAE_3, model_path_SAE_3)
    
final_net3 = torch.load(model_path_SAE_3)
_,h3 = final_net3(h2)
para_dict['AE3_weight']=final_net3.state_dict()['mlp1.0.weight']
para_dict['AE3_bias']=final_net3.state_dict()['mlp1.0.bias']
del train_data
del train_loader

Min_value = 1000

N_D = 32
N_Output =2
# sae_dnn = SAE_DNN(N_INPUT1,N_HIDDEN_1,N_HIDDEN_2,N_HIDDEN_3,N_D,N_Output,para_dict.copy()).cuda()
sae_dnn = SAE_DNN(N_INPUT1,N_HIDDEN_1,N_HIDDEN_2,N_HIDDEN_3,N_D,N_Output,para_dict.copy()).to(device)

optimizer = torch.optim.Adam(sae_dnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()

train_data = Data.TensorDataset(train_x,train_y)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# val_x = val_x.cuda()
# val_y = val_y.cuda()
# test_x = test_x.cuda()
# test_y = test_y.cuda()

val_x = val_x.to(device)  # gpu转成cpu
val_y = val_y.to(device)
test_x = test_x.to(device)
test_y = test_y.to(device)

loss_record=[]

for epoch in range(EPOCH):
    sae_dnn.train()
    for step, (b_x,b_y) in enumerate(train_loader):
        # b_x = b_x.cuda()
        # b_y = b_y.cuda()

        b_x = b_x.to(device)  #[20,25,4],batchsize = 20
        b_y = b_y.to(device)  #[20,2]

        output = sae_dnn(b_x)
        # loss = loss_func(output,b_y)
        loss = torch.mean(torch.pow(torch.sum( torch.pow(output-b_y,2),1),0.5)/100)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
             
    sae_dnn.eval()
    val_output = sae_dnn(val_x)
    loc_err1 = torch.mean(torch.pow(torch.sum( torch.pow(val_output-val_y,2),1),0.5)/100)
    print('Epoch: ', epoch,'| train:  ', float(loss),'|  val: ',float(loc_err1))
    loss_record.append(loss)
    if float(loc_err1) <  Min_value:
        Min_value = float(loc_err1)
        torch.save(sae_dnn, model_path)

final_net = torch.load(model_path)
with torch.no_grad():
    val_output = final_net(val_x)
loc_err = torch.mean(torch.pow(torch.sum( torch.pow(val_output-val_y,2),1),0.5)/100)
print('localization error for val data: %f'%loc_err)

with torch.no_grad():
    test_output = final_net(test_x)
loc_err1 = torch.mean(torch.pow(torch.sum( torch.pow(test_output-test_y,2),1),0.5)/100)
print('localization error for test data: %f'%loc_err1)
plt.plot(loss_record)

re_save = (torch.pow(torch.sum( torch.pow(test_output-test_y,2),1),0.5)/100).cpu().numpy()
save_path = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\SAE_DNN'+str(std)+'_ROOM'+str(ROOM)+'.pkl'
with open(save_path,'wb') as f:
    pickle.dump(re_save,f)

print('--------------------room is  %d  --------------------------'%(ROOM))
print('--------------------data set is  %d  --------------------------'%(data_set))