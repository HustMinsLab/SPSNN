# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:21:17 2020

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


EPOCH = 1000        # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 32

LR = 0.001

Min_value = 1000

path = '..\\pickleData\\raw_ratio\\pics\\2D_CNN\\train_room_'+ str(ROOM)+'_disturb'+str(std)+'.pkl'
path1 = '..\\pickleData\\raw_ratio\\pics\\2D_CNN\\val_room_'+ str(ROOM)+'.pkl'
path2 = '..\\pickleData\\raw_ratio\\pics\\2D_CNN\\test_room_'+ str(ROOM)+'.pkl'

model_path ='..\\model\\2D-CNN\\data_set_'+str(data_set)+'\\disturbed\\2DCNN'+str(std)+'_Model For  Room_'+str(ROOM)+'.pkl'

with open(path,'rb') as f:
    picture_data = pickle.load(f)

class CNN1(nn.Module):
    def __init__(self):
        super(CNN1,self).__init__()
        self.conv1 = nn.Sequential( #input shape(1,18,18)
                nn.Conv2d(
                        in_channels=1, # input height
                        out_channels=64,  # n_filters
                        kernel_size=3,   # filter size
                        stride=1,       # filter movement/step
                        padding=1      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                        ),     # output shape (64, 18,18)
                nn.ReLU(),
                )
        self.conv2 = nn.Sequential(     # input shape (64, 18,18)
                nn.Conv2d(64,64,3,1,1), # output shape (64, 9, 9)
                nn.ReLU(),  
                nn.MaxPool2d(2,padding = 0)
                )
        self.conv3 = nn.Sequential(     # input shape (64, 9, 9)
                nn.Conv2d(64,128,3,1,1), # output shape (128, 9, 9)
                nn.ReLU(),  
                )
        self.conv4 = nn.Sequential(     # input shape (128, 9, 9)
                nn.Conv2d(128,128,3,1,1), # output shape (128, 5, 5)
                nn.ReLU(),  
                nn.MaxPool2d(2,padding = 1)
                )
        self.mlp1 = nn.Sequential(
                torch.nn.Linear(128* 5 * 5,512),
#                torch.nn.Dropout(0.5),
                nn.ReLU()
                )
        self.mlp2 = nn.Sequential(
                torch.nn.Linear(512,512),
#                torch.nn.Dropout(0.5),
                nn.ReLU()
                )
        
        self.out = nn.Linear(512, 2)   # fully connected layer, output 2 classes
    
    
    def forward(self,x):
        x = x.view(x.size()[0],1,x.size()[1],x.size()[2])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0),-1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        x = self.mlp1(x)
        x = self.mlp2(x)
        output = self.out(x)
        return output
class CNN2(nn.Module):
    def __init__(self):
        super(CNN2,self).__init__()
        self.conv1 = nn.Sequential( #input shape(1,19,19)
                nn.Conv2d(
                        in_channels=1, # input height
                        out_channels=64,  # n_filters
                        kernel_size=3,   # filter size
                        stride=1,       # filter movement/step
                        padding=1      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                        ),     # output shape (64, 19,19)
                nn.ReLU(),
                )
        self.conv2 = nn.Sequential(     # input shape (64, 19,19)
                nn.Conv2d(64,64,3,1,1), # output shape (64, 10, 10)
                nn.ReLU(),  
                nn.MaxPool2d(2,padding = 1)
                )
        self.conv3 = nn.Sequential(     # input shape (64, 10, 10)
                nn.Conv2d(64,128,3,1,1), # output shape (128, 10, 10)
                nn.ReLU(),  
                )
        self.conv4 = nn.Sequential(     # input shape (128, 10, 10)
                nn.Conv2d(128,128,3,1,1), # output shape (128, 5, 5)
                nn.ReLU(),  
                nn.MaxPool2d(2,padding = 0)
                )
        self.mlp1 = nn.Sequential(
                torch.nn.Linear(128* 5 * 5,512),
#                torch.nn.Dropout(0.5),
                nn.ReLU()
                )
        self.mlp2 = nn.Sequential(
                torch.nn.Linear(512,512),
#                torch.nn.Dropout(0.5),
                nn.ReLU()
                )
        
        self.out = nn.Linear(512, 2)   # fully connected layer, output 2 classes
    
    
    def forward(self,x):
        x = x.view(x.size()[0],1,x.size()[1],x.size()[2])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0),-1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        x = self.mlp1(x)
        x = self.mlp2(x)
        output = self.out(x)
        return output
class CNN3(nn.Module):
    def __init__(self):
        super(CNN3,self).__init__()
        self.conv1 = nn.Sequential( #input shape(1,20,20)
                nn.Conv2d(
                        in_channels=1, # input height
                        out_channels=64,  # n_filters
                        kernel_size=3,   # filter size
                        stride=1,       # filter movement/step
                        padding=1      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                        ),     # output shape (64, 20, 20)
                nn.ReLU(),
                )
        self.conv2 = nn.Sequential(     # input shape (64, 20, 20)
                nn.Conv2d(64,64,3,1,1), # output shape (64, 10, 10)
                nn.ReLU(),  
                nn.MaxPool2d(2,padding =0)
                )
        self.conv3 = nn.Sequential(     # input shape (64, 10, 10)
                nn.Conv2d(64,128,3,1,1), # output shape (128, 10, 10)
                nn.ReLU(),  
                )
        self.conv4 = nn.Sequential(     # input shape (128, 10, 10)
                nn.Conv2d(128,128,3,1,1), # output shape (128, 5, 5)
                nn.ReLU(),  
                nn.MaxPool2d(2,padding = 0)
                )
        self.mlp1 = nn.Sequential(
                torch.nn.Linear(128* 5 * 5,512),
                # torch.nn.Dropout(0.5),
                nn.ReLU()
                )
        self.mlp2 = nn.Sequential(
                torch.nn.Linear(512,512),
                # torch.nn.Dropout(0.5),
                nn.ReLU()
                )
        
        self.out = nn.Linear(512, 2)   # fully connected layer, output 2 classes
    
    
    def forward(self,x):
        x = x.view(x.size()[0],1,x.size()[1],x.size()[2])  # (32,1,19,19)
        x = self.conv1(x)  # (32,64,19,19)
        x = self.conv2(x)  # (32,64,9,9)
        x = self.conv3(x)  # (32,128,9,9)
        x = self.conv4(x)  # (32,128,4,4)
        x = x.view(x.size(0),-1)   # 展平多维的卷积图成 (batch_size, 128 * 4 * 4)
        x = self.mlp1(x)
        x = self.mlp2(x)
        output = self.out(x)
        return output

class CNN4(nn.Module):
    def __init__(self):
        super(CNN4,self).__init__()
        self.conv1 = nn.Sequential( #input shape(1,21,21)
                nn.Conv2d(
                        in_channels=1, # input height
                        out_channels=64,  # n_filters
                        kernel_size=3,   # filter size
                        stride=1,       # filter movement/step
                        padding=1      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                        ),     # output shape (64, 21, 21)
                nn.ReLU(),
                )
        self.conv2 = nn.Sequential(     # input shape (64, 21, 21)
                nn.Conv2d(64,64,3,1,1), # output shape (64, 11, 11)
                nn.ReLU(),  
                nn.MaxPool2d(2,padding =1)
                )
        self.conv3 = nn.Sequential(     # input shape (64, 11, 11)
                nn.Conv2d(64,128,3,1,1), # output shape (128, 11, 11)
                nn.ReLU(),  
                )
        self.conv4 = nn.Sequential(     # input shape (128, 11, 11)
                nn.Conv2d(128,128,3,1,1), # output shape (128, 6, 6)
                nn.ReLU(),  
                nn.MaxPool2d(2,padding =1)
                )
        self.mlp1 = nn.Sequential(
                torch.nn.Linear(128* 6 * 6,512),
#                torch.nn.Dropout(0.5),
                nn.ReLU()
                )
        self.mlp2 = nn.Sequential(
                torch.nn.Linear(512,512),
#                torch.nn.Dropout(0.5),
                nn.ReLU()
                )
        
        self.out = nn.Linear(512, 2)   # fully connected layer, output 2 classes
    
    
    def forward(self,x):
        x = x.view(x.size()[0],1,x.size()[1],x.size()[2])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0),-1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        x = self.mlp1(x)
        x = self.mlp2(x)
        output = self.out(x)
        return output

Net_dict={1:CNN1(),2:CNN1(),3:CNN2(),4:CNN3()}  # room1:18*18;room2:18*18;room3:19*19;room4:20*20

# cnn = Net_dict[ROOM].cuda()
cnn = Net_dict[ROOM].to(device)




#训练集
x,y,z,k = len(picture_data),(picture_data[0][0]).shape[0],(picture_data[0][0]).shape[1],len(picture_data[0][1])
train_x = torch.zeros((x,y,z))
train_y = torch.zeros((x,k)) 

for i in range(x): 
    train_x[i] = torch.from_numpy(picture_data[i][0])  
    train_y[i] = torch.Tensor(picture_data[i][1]) 

    
with open(path1,'rb') as f:
    picture_data_val = pickle.load(f)
with open(path2,'rb') as f:
    picture_data_test = pickle.load(f)

x,y,z,k = len(picture_data_val),(picture_data_val[0][0]).shape[0],(picture_data_val[0][0]).shape[1],len(picture_data_val[0][1])
val_x = torch.zeros((x,y,z))
val_y = torch.zeros((x,k)) 

for i in range(x):
     
    val_x[i] = torch.from_numpy(picture_data_val[i][0])  
    val_y[i] = torch.Tensor(picture_data_val[i][1]) 


x,y,z,k = len(picture_data_test),(picture_data_test[0][0]).shape[0],(picture_data_test[0][0]).shape[1],len(picture_data_test[0][1])
test_x = torch.zeros((x,y,z))
test_y = torch.zeros((x,k)) 

for i in range(x):
    test_x[i] = torch.from_numpy(picture_data_test[i][0])  
    test_y[i] = torch.Tensor(picture_data_test[i][1]) 



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

#训练
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()

#model_path = '.\\model\\Model For Room_'+str(ROOM)+'.pth'
#model_path = '.\\model\\for_adjust_var\\Model For Room_'+str(ROOM)+'.pth'
#model_path = '.\\model\\ratio_data\\Model For Room_'+str(ROOM)+'.pth'
loss_record=[]
for epoch in range(EPOCH):
    cnn.train()
    for step, (b_x,b_y) in enumerate(train_loader):
        # b_x = b_x.cuda()
        # b_y = b_y.cuda()

        b_x = b_x.to(device)  # [20,25,4],batchsize = 20
        b_y = b_y.to(device)  # [20,2]
        output = cnn(b_x)
        # loss = loss_func(output,b_y)
        loss = torch.mean(torch.pow(torch.sum( torch.pow(output-b_y,2),1),0.5)/100) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loc_err = torch.mean(torch.pow(torch.sum( torch.pow(output-b_y,2),1),0.5)/100)       
#        print('Epoch: ', epoch, '| Step: ', step, 'loc_err:', loc_err)
    cnn.eval()
    val_output = cnn(val_x)
    loc_err1 = torch.mean(torch.pow(torch.sum( torch.pow(val_output-val_y,2),1),0.5)/100)
#    print('Epoch: ', epoch,'| localization error for testdata:  ', loc_err)
    print('Epoch: ', epoch,'| train:  ', float(loc_err),'|  val: ',float(loc_err1))
    loss_record.append(float(loss))
    if float(loc_err1) <  Min_value:
        Min_value = float(loc_err1)
        torch.save(cnn, model_path)





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
save_path = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\2DCNN'+str(std)+'_ROOM'+str(ROOM)+'.pkl'
with open(save_path,'wb') as f:
    pickle.dump(re_save,f)

print('--------------------room is  %d  --------------------------'%(ROOM))
print('--------------------data set is  %d  --------------------------'%(data_set))