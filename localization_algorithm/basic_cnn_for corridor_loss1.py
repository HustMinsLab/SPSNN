# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:41:34 2020

@author: DELL PC
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 10:11:14 2020

@author: DELL PC
训练时左边输入信号空间欧氏距离图，加上右边的物理空间图的约束，loss = lossF+lossPre
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
import pickle
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 无cuda,使用cpu
seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

ROOM = 2
data_set = 1
ERROR = 2
Min_value = 100
std = 250

EPOCH = 2000           # 训练整批数据多少次
BATCH_SIZE = 20  # 32
LR = 0.00093



def getPhyPicture(cor, graph):    
    result = np.zeros((len(graph), len(graph[0])), dtype = float)
    for i in range(len(graph)):
        for j in range(len(graph[0])):
            x1, y1, x2, y2 = cor[0], cor[1], graph[i][j][0],graph[i][j][1]
            deltaV = np.array([x1-x2,y1-y2])
            result[i,j] = np.linalg.norm(deltaV,ord =2)
             
    return result



#----------------------------------------------------final_mix--------------------------------
path1 = '..\\pickleData\\final_mix\\ave\\train_data_ROOM_'+str(ROOM)+'.pkl'
# path1 = '..\\pickleData\\raw_ratio\\average_data\\room_'+str(ROOM)+'_train.pkl'

with open(path1,'rb') as f:
    ratio_train = pickle.load(f)


max_x,min_x,max_y,min_y = ratio_train.loc[:,"xLabel"].max() + 30, ratio_train.loc[:,"xLabel"].min() - 30,ratio_train.loc[:,"yLabel"].max() + 30, ratio_train.loc[:,"yLabel"].min() - 30
# max_x,min_x,max_y,min_y = ratio_train.loc[:,"xLabel"].max(), ratio_train.loc[:,"xLabel"].min(),ratio_train.loc[:,"yLabel"].max(), ratio_train.loc[:,"yLabel"].min()

if min_x< 0: min_x = 0
if min_y<0 : min_y = 0
del ratio_train

path = '..\\pickleData\\raw_ratio\\pics\\06\\disturbed\\train_room_'+ str(ROOM)+'_disturb'+ str(std)+'.pkl'  # 数据集1，扰动数据
path1 = '..\\pickleData\\raw_ratio\\pics\\06\\disturbed\\val_room_'+ str(ROOM)+'_disturb'+ str(std)+'.pkl'
path2 = '..\\pickleData\\raw_ratio\\pics\\06\\disturbed\\test_room_'+ str(ROOM)+'_disturb'+ str(std)+'.pkl'
model_path = '..\\model\\CNNLoss\\data_set_'+str(data_set)+'\\06\\L_Deeper_Model For Room_'+str(ROOM)+'_disturb'+str(std)+'.pth'

with open(path,'rb') as f:
    picture_data = pickle.load(f)
with open(path1,'rb') as f:
    picture_data_val = pickle.load(f)
with open(path2,'rb') as f:
    picture_data_test = pickle.load(f)


train_size = len(picture_data)
val_size = len(picture_data_val)
test_size = len(picture_data_test)



path4 = '..\\pickleData\\different_grid\\06\\room'+str(ROOM)+'_06.pkl'
with open(path4,'rb') as f:
    graph_06=pickle.load(f)


#重新构造成对输入的样本
#训练样本
picture_data_train_new = [[None,None] for x in range(train_size)]

for  i in range(train_size):
    #构造物理空间图片
    output = getPhyPicture( picture_data[i][1],graph_06 )    
    #赋值
    picture_data_train_new[i][0] = np.array([picture_data[i][0],output])
    picture_data_train_new[i][1] = np.array([picture_data[i][1],picture_data[i][1]])              #标签为多余标签




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
        self.mlp2 = nn.Sequential(
                torch.nn.Linear(100,5),
#                torch.nn.Dropout(0.5),
                nn.ReLU()
                )
        
        self.out = nn.Linear(5, 2)   # fully connected layer, output 2 classes
    
    
    def forward(self,x):
        x = x.view(x.size()[0],1,x.size()[1],x.size()[2])
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)   # 展平多维的卷积图成 (batch_size, 32 * 5 * 4)
        x = self.mlp1(x)
        x1 = self.mlp2(x)
        # print(x1)
        output = self.out(x1)
        # print("output",output)
        return output,x1



Net_dict={1:CNN(),2:CNN(),3:CNN(),4:CNN()}
# cnn = Net_dict[ROOM].cuda()
cnn = Net_dict[ROOM].to(device)


#构造输入数据、输出数据



#train_data = [......]  #此处修改输入
# x:batch,t:,h:height,w:width,c:channel
x,t,h,w,c = len(picture_data_train_new),len(picture_data_train_new[0]),(picture_data_train_new[0][0][0]).shape[0],(picture_data_train_new[0][0][0]).shape[1],len(picture_data_train_new[0][1][0])
train_x = torch.zeros((x,t,h,w))
train_y = torch.zeros((x,t,c)) 
for i in range(x):
    temp1 = picture_data_train_new[i][0][0].copy()
    temp2 = picture_data_train_new[i][0][1].copy()
    #标准化
    if np.std(temp1) != 0:  
        picture_data_1 = (temp1-np.mean(temp1))/np.std(temp1)
    else:
        picture_data_1 = (temp1-np.mean(temp1))
    if np.std(temp2) != 0:  
        picture_data_2 = (temp2-np.mean(temp2))/np.std(temp2)
    else:
        picture_data_2 = (temp2-np.mean(temp2))
    train_x[i] = torch.from_numpy(np.array([picture_data_1,picture_data_2])) 
    temp3 = picture_data_train_new[i][1].copy()
    temp3[:,0] = (temp3[:,0] -min_x )/(max_x-min_x)
    temp3[:,1] = (temp3[:,1] -min_y )/(max_y-min_y)
    train_y[i] = torch.from_numpy(temp3)  
    del temp1
    del temp2
    del temp3
    del picture_data_1
    del picture_data_2



x,y,z,k = len(picture_data_val),(picture_data_val[0][0]).shape[0],(picture_data_val[0][0]).shape[1],len(picture_data_val[0][1])
val_x = torch.zeros((x,y,z))
val_y = torch.zeros((x,k)) 
for i in range(x):
    temp = picture_data_val[i][0].copy()
    #标准化
    picture_data_1 = (temp-np.mean(temp))/np.std(temp)     
    val_x[i] = torch.from_numpy(picture_data_1) 
    val_y[i] = torch.Tensor(picture_data_val[i][1])  
    del temp
    del picture_data_1


x,y,z,k = len(picture_data_test),(picture_data_test[0][0]).shape[0],(picture_data_test[0][0]).shape[1],len(picture_data_test[0][1])
test_x = torch.zeros((x,y,z))
test_y = torch.zeros((x,k)) 
for i in range(x):
    temp = picture_data_test[i][0].copy()
    #标准化
    picture_data_1 = (temp-np.mean(temp))/np.std(temp)     
    test_x[i] = torch.from_numpy(picture_data_1) 
    test_y[i] = torch.Tensor(picture_data_test[i][1])  
    del temp
    del picture_data_1
   

train_data = Data.TensorDataset(train_x,train_y)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


# val_x = val_x.cuda()
# val_y = val_y.cuda()
# test_x = test_x.cuda()
# test_y = test_y.cuda()

train_y = train_y.to(device)
val_x = val_x.to(device)  # gpu转成cpu
val_y = val_y.to(device)
test_x = test_x.to(device)
test_y = test_y.to(device)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=0.001)   # optimize all cnn parameters
loss_record=[]
loss_func = nn.MSELoss()  # 损失函数
for epoch in range(EPOCH):
    cnn.train()
    for step, (b_x,b_y) in enumerate(train_loader):
        # b_x1 = b_x[:,0,:,:].cuda()
        # b_x2 = b_x[:,1,:,:].cuda()
        # b_y1 = b_y[:,0,:].cuda()

        b_x1 = b_x[:, 0, :, :].to(device)
        b_x2 = b_x[:, 1, :, :].to(device)
        b_y1 = b_y[:, 0, :].to(device)
        output1,v1 = cnn(b_x1)
        output2,v2 = cnn(b_x2)
        # loss1 = torch.mean(torch.pow(torch.sum( torch.pow(output1-b_y1,2),1),0.5)/100)
        # loss2 = torch.mean(torch.pow(torch.sum( torch.pow(output2-b_y2,2),1),0.5)/100)

        output1[:, 0] = output1[:, 0] * (max_x - min_x) + min_x
        output1[:, 1] = output1[:, 1] * (max_y - min_y) + min_y

        output2[:, 0] = output2[:, 0] * (max_x - min_x) + min_x
        output2[:, 1] = output2[:, 1] * (max_y - min_y) + min_y

        b_y1[:, 0] = b_y1[:, 0] * (max_x - min_x) + min_x
        b_y1[:, 1] = b_y1[:, 1] * (max_y - min_y) + min_y


        lossPre = torch.mean(torch.pow(torch.sum(torch.pow(output1 - b_y1, 2), 1), 0.5) / 100)  # error2
        lossF = loss_func(v1, v2)  # error2

        loss = lossPre + lossF

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    cnn.eval()
    val_output,_ = cnn(val_x)

    val_output[:,0] = val_output[:,0]*(max_x-min_x) + min_x
    val_output[:,1] = val_output[:,1]*(max_y-min_y) + min_y
    # loc_err1 = torch.mean(torch.sum(torch.pow(val_output - val_y, 2), 1))
    loc_err1 = torch.mean(torch.pow(torch.sum( torch.pow(val_output-val_y,2),1),0.5)/100)
        
    # print('Epoch: ', epoch,'|  loss: ',float(loss), '|  val: ',float(loc_err1))
    print('Epoch: ', epoch,'|  loss: ',int(loss),'|  lossPre: ',float(lossPre),'|  lossF: ',float(lossF),'|  val: ',float(loc_err1))
    loss_record.append(loss)
    if float(loc_err1) <  Min_value:
        Min_value = float(loc_err1)
        torch.save(cnn, model_path)

final_net = torch.load(model_path)
with torch.no_grad():
    val_output,_ = final_net(val_x)
    val_output[:,0] = val_output[:,0]*(max_x-min_x) + min_x
    val_output[:,1] = val_output[:,1]*(max_y-min_y) + min_y

loc_err = torch.mean(torch.pow(torch.sum( torch.pow(val_output-val_y,2),1),0.5)/100)
print('localization error for val data: %f'%loc_err)

with torch.no_grad():
    test_output,_ = final_net(test_x)
    test_output[:,0] = test_output[:,0]*(max_x-min_x) + min_x
    test_output[:,1] = test_output[:,1]*(max_y-min_y) + min_y
loc_err1 = torch.mean(torch.pow(torch.sum( torch.pow(test_output-test_y,2),1),0.5)/100)
print('localization error for test data: %f'%loc_err1)
plt.plot(loss_record)

re_save = (torch.pow(torch.sum( torch.pow(test_output-test_y,2),1),0.5)/100).cpu().numpy()

save_path = '..\\pickleData\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\CNNLoss_'+ str(std)+'_ROOM'+str(ROOM)+'.pkl'
with open(save_path,'wb') as f:
    pickle.dump(re_save,f)

print('--------------------room is  %d  --------------------------'%(ROOM))
print('--------------------data set is  %d  --------------------------'%(data_set))