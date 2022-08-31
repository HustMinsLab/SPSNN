# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:43:06 2019

@author: DELL PC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
import pickle
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 无cuda,使用cpu
seed = 1
torch.manual_seed(seed)  # 固定随机数
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

ROOM = 4
data_set = 1
ERROR = 2
Min_value = 100
std = 250
#percentage = 0.1

#K = 17*13       #Kmeans的K
#K = 1000
#tag = 7


path = '..\\pickleData\\raw_ratio\\pics\\06\\disturbed\\train_room_'+ str(ROOM)+'_disturb'+ str(std)+'.pkl'  # 数据集1，扰动数据
path1 = '..\\pickleData\\raw_ratio\\pics\\06\\disturbed\\val_room_'+ str(ROOM)+'_disturb'+ str(std)+'.pkl'
path2 = '..\\pickleData\\raw_ratio\\pics\\06\\disturbed\\test_room_'+ str(ROOM)+'_disturb'+ str(std)+'.pkl'
model_path = '..\\model\\CNNEu\\data_set_'+str(data_set)+'\\06\\Model For Room_'+str(ROOM)+'_disturb'+str(std)+'.pth'

with open(path,'rb') as f:  # rb：打开二进制文件，r：打开文本
    picture_data = pickle.load(f)


EPOCH = 2000          # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 32
LR = 0.045


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential( #input shape(1,17,13)
                nn.Conv2d(
                        in_channels=1, # input height
                        out_channels=16,  # n_filters
                        kernel_size=5,   # filter size
                        stride=1,       # filter movement/step
                        padding=2      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                        ),     # output shape (16, 17, 13)
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,padding = 1),  # 在 2x2 空间里向下采样, output shape (16, 9, 7)
                )
        self.conv2 = nn.Sequential(     # input shape (16, 9, 7)
                nn.Conv2d(16,32,5,1,2), # output shape (32, 9, 7)
                nn.ReLU(),  
                nn.MaxPool2d(2,padding = 1),    # output shape (32, 5, 4)
                )
        self.mlp1 = nn.Sequential(
                torch.nn.Linear(32 * 5 * 4,100),
#                torch.nn.Dropout(0.5),
                nn.ReLU()
                )
        
        self.out = nn.Linear(100, 2)   # fully connected layer, output 2 classes
    
    
    def forward(self,x):
        x = x.view(x.size()[0],1,x.size()[1],x.size()[2])
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)  (其中-1表示不确定，肯定是batch_size行，但是列数不确定)
        x = self.mlp1(x)
        output = self.out(x)
        return output

class CNN5(nn.Module):
    def __init__(self):
        super(CNN5,self).__init__()
        self.conv1 = nn.Sequential( #input shape(1,18,4)
                nn.Conv2d(
                        in_channels=1, # input height
                        out_channels=16,  # n_filters
                        kernel_size=5,   # filter size
                        stride=1,       # filter movement/step
                        padding=2      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                        ),     # output shape (16, 18, 4)
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,padding = (0,0)),  # 在 2x2 空间里向下采样, output shape (16, 9, 2)
                )
        self.conv2 = nn.Sequential(     # input shape (16, 9, 2)
                nn.Conv2d(16,32,5,1,2), # output shape (32, 9, 2)
                nn.ReLU(),  
                nn.MaxPool2d(2,padding = (1,0)),    # output shape (32, 5, 1)
                )
        self.mlp1 = nn.Sequential(
                torch.nn.Linear(32 * 5 * 1,100),
#                torch.nn.Dropout(0.5),
                nn.ReLU()
                )
        
        self.out = nn.Linear(100, 2)   # fully connected layer, output 2 classes
    
    
    def forward(self,x):
        x = x.view(x.size()[0],1,x.size()[1],x.size()[2])
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        x = self.mlp1(x)
        output = self.out(x)
        return output



class CNN6(nn.Module):
    def __init__(self):
        super(CNN6,self).__init__()
        self.conv1 = nn.Sequential( #input shape(1,25,29)
                nn.Conv2d(
                        in_channels=1, # input height
                        out_channels=16,  # n_filters
                        kernel_size=5,   # filter size
                        stride=1,       # filter movement/step
                        padding=2      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                        ),     # output shape (16, 25, 29)
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,padding = 1),  # 在 2x2 空间里向下采样, output shape (16, 13, 15)
                )
        self.conv2 = nn.Sequential(     # input shape (16, 13, 15)
                nn.Conv2d(16,32,5,1,2), # output shape (32, 13, 15)
                nn.ReLU(),  
                nn.MaxPool2d(2,padding = 1),    # output shape (32, 7, 8)
                )
#        self.conv3 = nn.Sequential(     # input shape (32, 7, 8)
#                nn.Conv2d(32,32,5,1,2), # output shape (32, 7, 8)
#                nn.ReLU(),  
#                nn.MaxPool2d(2,padding = (1,0)),    # output shape (32,4, 4)
#                )
        self.mlp1 = nn.Sequential(
                torch.nn.Linear(32 * 7 * 8,100),
#                torch.nn.Dropout(0.5),
                nn.ReLU()
                )
        
        self.out = nn.Linear(100, 2)   # fully connected layer, output 2 classes
    
    
    def forward(self,x):
        x = x.view(x.size()[0],1,x.size()[1],x.size()[2])
        x = self.conv1(x)
        x = self.conv2(x)
#        x = self.conv3(x)
        x = x.view(x.size(0),-1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        x = self.mlp1(x)
        output = self.out(x)
        return output

class CNN8(nn.Module):
    def __init__(self):
        super(CNN8,self).__init__()
        self.conv1 = nn.Sequential( #input shape(1,28,4)
                nn.Conv2d(
                        in_channels=1, # input height
                        out_channels=16,  # n_filters
                        kernel_size=5,   # filter size
                        stride=1,       # filter movement/step
                        padding=2      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                        ),     # output shape (16, 28,4)
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2,padding = (0,0)),  # 在 2x2 空间里向下采样, output shape (16, 14, 2)
                )
        self.conv2 = nn.Sequential(     # input shape (16, 14, 2)
                nn.Conv2d(16,32,5,1,2), # output shape (32, 14, 2)
                nn.ReLU(),  
                nn.MaxPool2d(2,padding = (0,0)),    # output shape (32, 7, 1)
                )
        self.mlp1 = nn.Sequential(
                torch.nn.Linear(32 * 7 * 1,100),
#                torch.nn.Dropout(0.5),
                nn.ReLU()
                )
        
        self.out = nn.Linear(100, 2)   # fully connected layer, output 2 classes
    
    
    def forward(self,x):
        x = x.view(x.size()[0],1,x.size()[1],x.size()[2])
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        x = self.mlp1(x)
        output = self.out(x)
        return output

class CNN_deep(nn.Module):
    def __init__(self):
        super(CNN_deep,self).__init__()
        self.conv1 = nn.Sequential( #input shape(1,25,29)
                nn.Conv2d(
                        in_channels=1, # input height
                        out_channels=16,  # n_filters
                        kernel_size=5,   # filter size
                        stride=1,       # filter movement/step
                        padding=2      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
                        ),     # output shape (16, 25, 29)
                nn.ReLU(),
#                nn.MaxPool2d(kernel_size=2,padding = 1),  # 在 2x2 空间里向下采样, output shape (16, 13, 15)
                )
        self.conv2 = nn.Sequential(     # input shape (16, 25, 29)
                nn.Conv2d(16,32,5,1,2), # output shape (32, 25, 29)
                nn.ReLU(),  
#                nn.MaxPool2d(2,padding = 1),    # output shape (32, 7, 8)
                )
#        self.conv3 = nn.Sequential(     # input shape (32, 25, 29)
#                nn.Conv2d(32,32,5,1,2), # output shape (32, 25, 29)
#                nn.ReLU(),  
##                nn.MaxPool2d(2,padding = (1,0)),    # output shape (32,4, 4)
#                )
        self.mlp1 = nn.Sequential(
                torch.nn.Linear(32 * 25 * 29,100),
#                torch.nn.Dropout(0.5),
                nn.ReLU()
                )
        self.mlp2 = nn.Sequential(
                torch.nn.Linear(100,50),
#                torch.nn.Dropout(0.5),
                nn.ReLU()
        )
        
        self.out = nn.Linear(50, 2)   # fully connected layer, output 2 classes
    
    
    def forward(self,x):
        x = x.view(x.size()[0],1,x.size()[1],x.size()[2])
        x = self.conv1(x)
        x = self.conv2(x)
#        x = self.conv3(x)
        x = x.view(x.size(0),-1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        x = self.mlp1(x)
        x = self.mlp2(x)
        output = self.out(x)
        return output

Net_dict={1:CNN(),2:CNN(),3:CNN(),4:CNN(),5:CNN5(),6:CNN6(),7:CNN5(),8:CNN8(),9:CNN8()}

# cnn = Net_dict[ROOM].cuda()
cnn = Net_dict[ROOM].to(device)



#构造输入数据、输出数据






#train_data = [......]  #此处修改输入
x,y,z,k = len(picture_data),(picture_data[0][0]).shape[0],(picture_data[0][0]).shape[1],len(picture_data[0][1])
train_x = torch.zeros((x,y,z))  # 指纹(x:横坐标，y:纵坐标，z:RSS指纹)
train_y = torch.zeros((x,k)) # 标签
for i in range(x):
    temp = picture_data[i][0]
    #标准化
    if np.std(temp) == 0:
        picture_data_1 = temp-np.mean(temp)
    else:
        picture_data_1 = (temp-np.mean(temp))/np.std(temp)
    train_x[i] = torch.from_numpy(picture_data_1)     
    train_y[i] = torch.Tensor(picture_data[i][1])  
    del temp
    del picture_data_1
#将数据分开为图像和标签，转换为tensor格式

with open(path1,'rb') as f:
    picture_data_val = pickle.load(f)
with open(path2,'rb') as f:
    picture_data_test = pickle.load(f)

x,y,z,k = len(picture_data_val),(picture_data_val[0][0]).shape[0],(picture_data_val[0][0]).shape[1],len(picture_data_val[0][1])
val_x = torch.zeros((x,y,z))
val_y = torch.zeros((x,k)) 
for i in range(x):
    temp = picture_data_val[i][0]
    #标准化
    if np.std(temp) == 0:
        picture_data_1 = temp-np.mean(temp)
    else:
        picture_data_1 = (temp-np.mean(temp))/np.std(temp)     
    val_x[i] = torch.from_numpy(picture_data_1) 
    val_y[i] = torch.Tensor(picture_data_val[i][1])  
    del temp
    del picture_data_1


x,y,z,k = len(picture_data_test),(picture_data_test[0][0]).shape[0],(picture_data_test[0][0]).shape[1],len(picture_data_test[0][1])
test_x = torch.zeros((x,y,z))
test_y = torch.zeros((x,k)) 
for i in range(x):
    temp = picture_data_test[i][0]
    #标准化
    if np.std(temp) == 0:
        picture_data_1 = temp-np.mean(temp)
    else:
        picture_data_1 = (temp-np.mean(temp))/np.std(temp)     
    test_x[i] = torch.from_numpy(picture_data_1) 
    test_y[i] = torch.Tensor(picture_data_test[i][1])  
    del temp
    del picture_data_1

train_data = Data.TensorDataset(train_x,train_y)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)  #数据封装


# val_x = val_x.cuda()  # cpu转成gpu
# val_y = val_y.cuda()
# test_x = test_x.cuda()
# test_y = test_y.cuda()

val_x = val_x.to(device)  # gpu转成cpu
val_y = val_y.to(device)
test_x = test_x.to(device)
test_y = test_y.to(device)

#训练
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters，优化
loss_func = nn.MSELoss()  # 损失函数

for epoch in range(EPOCH):  # 训练次数
    cnn.train()
    for step, (b_x,b_y) in enumerate(train_loader):
        # b_x = b_x.cuda()
        # b_y = b_y.cuda()

        b_x = b_x.to(device)  #[20,25,4],batchsize = 20
        b_y = b_y.to(device)  #[20,2]
        output = cnn(b_x)
        loss =  torch.mean(torch.pow(torch.sum( torch.pow(output-b_y,2),1),0.5)/100)  # error_2
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

re_save = (torch.pow(torch.sum( torch.pow(test_output-test_y,2),1),0.5)/100).cpu().numpy()
save_path = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\CNNEu_'+ str(std)+'_ROOM'+str(ROOM)+'.pkl'
with open(save_path,'wb') as f:
    pickle.dump(re_save,f)

print('--------------------room is  %d  --------------------------'%(ROOM))
print('--------------------data set is  %d  --------------------------'%(data_set))