# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:26:38 2019

@author: DELL PC
"""
import numpy as np
import pickle
import pandas as pd

ROOM = 4
data_set = 1
K_value = 1
interval = '06'
std = 250


def find_the_shortest_grid(x,y,graph_06):
    res =[0,0]
    shortest = 100000
    for i in range(len(graph_06)):
        for j in range(len(graph_06[0])):
            dis = np.linalg.norm(np.array([x,y])-np.array(graph_06[i][j]))
            if  dis < shortest:
                shortest =dis
                res = [i,j]
    return res
        
    
def localization(fpdb,graph,ratio_val,K):
    MAX = 10000
    res = np.zeros([ratio_val.shape[0],5])
    for i in range(ratio_val.shape[0]):
        test_f = list(ratio_val.iloc[i,1:761])
        res[i,0],res[i,1] = ratio_val.iloc[i,761]/100.0,ratio_val.iloc[i,762]/100.0
        dis = []
        for j in range(len(fpdb)):
            for k in range(len(fpdb[0])):
                if len(fpdb[j][k][1]) <= 0:
                    dis.append(MAX)
                else:
                    fb_f = fpdb[j][k][3]
                    count = 0
                    deltaSum = 0
                    for m in range(len(test_f)):
                        if test_f[m] != 0 and fb_f[m] != 0:
                            count += 1
                            deltaSum += (test_f[m] - fb_f[m])**2
                    if count == 0:
                        dis.append(MAX)
                    else:
                        dis.append(pow(deltaSum,0.5)/count)
        Mat1d = np.array(dis)
        Mat1d_order= np.sort(Mat1d)[:K]
        idxMaxid = np.zeros([len(Mat1d_order),])
        row = []
        col = []
        b = len(fpdb[0])
        for j in range(len(Mat1d_order)):
            idxMaxid[j] = np.where(Mat1d == Mat1d_order[j] )[0][0]
            row.append(int(idxMaxid[j]/b))
            col.append(int(idxMaxid[j]%b))
            Mat1d_order[j] = 1/(Mat1d_order[j]+0.001)
        Mat1d_order = Mat1d_order/np.sum(Mat1d_order)
        x1 = 0; y1 = 0;
        for j in range(len(Mat1d_order)):
            x1 = x1 + Mat1d_order[j]*(graph[row[j]][col[j]][0]/100)
            y1 = y1 + Mat1d_order[j]*(graph[row[j]][col[j]][1]/100)
        
        res[i,2], res[i,3] = x1, y1
    res[:,4] = np.sqrt( np.power(res[:,0]-res[:,2],2) + np.power(res[:,1]-res[:,3],2) )
    error = np.mean(res[:,4])
    return res, error

#导入数据
path1 = '..\\pickleData\\raw_ratio\\average_data\\disturbed_data\\room_'+str(ROOM)+'_disturb'+str(std)+'_train_disturbed.pkl'
path2 = '..\\pickleData\\raw_ratio\\average_data\\room_'+str(ROOM)+'_val.pkl'
path3 = '..\\pickleData\\raw_ratio\\average_data\\room_'+str(ROOM)+'_test.pkl'


with open(path1,'rb') as f:
    ratio_train = pickle.load(f)
with open(path2,'rb') as f:
    ratio_val = pickle.load(f)
with open(path3,'rb') as f:
    ratio_test = pickle.load(f)



#----------------取消验证集增加代码------------------------------------------
ratio_val=pd.concat([ratio_val,ratio_test],ignore_index=True)
# ---------------------------------------------------------------------

#------------------生成图片-------------------------------------------------
path4 = '..\\pickleData\\different_grid\\06\\room'+str(ROOM)+'_06.pkl'
path5 = '..\\pickleData\\different_grid\\10\\room'+str(ROOM)+'_10.pkl'
path6 = '..\\pickleData\\different_grid\\12\\room'+str(ROOM)+'_12.pkl'
with open(path4,'rb') as f:
    graph_06=pickle.load(f)
with open(path5,'rb') as f:
    graph_10=pickle.load(f)
with open(path6,'rb') as f:
    graph_12=pickle.load(f)
    
    
#构造栅格指纹
#0.6m
fpdb_06 = [[[[],[],[],[]] for col in range(len(graph_06[0]))] for row in range(len(graph_06))]
for i in range(len(ratio_train)):
    x , y = ratio_train.loc[i,'xLabel'],ratio_train.loc[i,'yLabel']
    target_grid = find_the_shortest_grid(x,y,graph_06)
    fpdb_06[target_grid[0]][target_grid[1]][0] = graph_06[target_grid[0]][target_grid[1]]
    fpdb_06[target_grid[0]][target_grid[1]][1].append(list(ratio_train.iloc[i,1:762]))
    fpdb_06[target_grid[0]][target_grid[1]][2].append([x,y])

for i in range(len(fpdb_06)):
    for j in range(len(fpdb_06[0])):
        if len(fpdb_06[i][j][1])>0:
            temp_res = []
            rss = np.array(fpdb_06[i][j][1])
            for k in range(rss.shape[1]):
                sumt = np.sum(rss[:,k])
                if sumt < 0:
                    temp_res.append(sumt/np.count_nonzero(rss[:,k]))
                else:
                    temp_res.append(0)      
            fpdb_06[i][j][3] = temp_res
        else:
            fpdb_06[i][j][3] = fpdb_06[i][j][1]

#1m
fpdb_10 = [[[[],[],[],[]] for col in range(len(graph_10[0]))] for row in range(len(graph_10))]
for i in range(len(ratio_train)):
    x , y = ratio_train.loc[i,'xLabel'],ratio_train.loc[i,'yLabel']
    target_grid = find_the_shortest_grid(x,y,graph_10)
    fpdb_10[target_grid[0]][target_grid[1]][0] = graph_10[target_grid[0]][target_grid[1]]
    fpdb_10[target_grid[0]][target_grid[1]][1].append(list(ratio_train.iloc[i,1:762]))
    fpdb_10[target_grid[0]][target_grid[1]][2].append([x,y])
for i in range(len(fpdb_10)):
    for j in range(len(fpdb_10[0])):
        if len(fpdb_10[i][j][1])>0:
            temp_res = []
            rss = np.array(fpdb_10[i][j][1])
            for k in range(rss.shape[1]):
                sumt = np.sum(rss[:,k])
                if sumt < 0:
                    temp_res.append(sumt/np.count_nonzero(rss[:,k]))
                else:
                    temp_res.append(0)      
            fpdb_10[i][j][3] = temp_res
        else:
            fpdb_10[i][j][3] = fpdb_10[i][j][1]

#1.2m
fpdb_12 = [[[[],[],[],[]] for col in range(len(graph_12[0]))] for row in range(len(graph_12))]
for i in range(len(ratio_train)):
    x , y = ratio_train.loc[i,'xLabel'],ratio_train.loc[i,'yLabel']
    target_grid = find_the_shortest_grid(x,y,graph_12)
    fpdb_12[target_grid[0]][target_grid[1]][0] = graph_12[target_grid[0]][target_grid[1]]
    fpdb_12[target_grid[0]][target_grid[1]][1].append(list(ratio_train.iloc[i,1:762]))
    fpdb_12[target_grid[0]][target_grid[1]][2].append([x,y])
for i in range(len(fpdb_12)):
    for j in range(len(fpdb_12[0])):
        if len(fpdb_12[i][j][1])>0:
            temp_res = []
            rss = np.array(fpdb_12[i][j][1])
            for k in range(rss.shape[1]):
                sumt = np.sum(rss[:,k])
                if sumt < 0:
                    temp_res.append(sumt/np.count_nonzero(rss[:,k]))
                else:
                    temp_res.append(0)      
            fpdb_12[i][j][3] = temp_res
        else:
            fpdb_12[i][j][3] = fpdb_12[i][j][1]


#---------------定位----------------------------
if interval == '06':
    fpdb = fpdb_06
    graph = graph_06
elif interval == '10':   
    fpdb = fpdb_10
    graph = graph_10
else:
    fpdb = fpdb_12
    graph = graph_12



val_error,val_mean_error = localization(fpdb,graph,ratio_val,K_value)
test_error,test_mean_error = localization(fpdb,graph,ratio_test,K_value)

print('val data mean_error of KNN when k = %d is %f (interval = %s, Room = %d)'%(K_value,val_mean_error,interval, ROOM))
print('test data mean_error of KNN when k = %d is %f (interval = %s, Room = %d)'%(K_value,test_mean_error,interval, ROOM))    

re_save_test = test_error[:,4]
re_save_val = val_error[:,4]
save_path_test = '..\\pickleData\\\localization_results\\data_set_'+str(data_set)+'\\disturbed\\NoTest_NN_ROOM'+str(ROOM)+'_disturb'+str(std)+'.pkl'

with open(save_path_test,'wb') as f:
    pickle.dump(re_save_test,f)