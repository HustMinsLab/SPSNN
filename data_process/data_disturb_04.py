# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 17:26:50 2019

@author: DELL PC

"""
# 扰动数据，std分布为0，50，100，150，200，250，300
# 构造0.6*0.6，1.0*1.0，1.2*1.2的图片矩阵
import numpy as np
import pickle
import math

ROOM = 4 # 1，2，3，4
data_set = 1
mean = 0 
std = 250  # 0，50，100，150，200，250，300
K_value = 1 # WKNN中K的取值，此处仅为观察扰动的幅度

def find_the_shortest_grid(x,y,graph_06):
    res =[0,0]
    shortest = 100000
    for i in range(len(graph_06)):  # 28行
        for j in range(len(graph_06[0])):  # 4列
            dis = np.linalg.norm(np.array([x,y])-np.array(graph_06[i][j]))  # 计算目标栅格和样本中坐标（x,y）的距离，坐标单位为厘米(cm)
            if  dis < shortest:
                shortest =dis
                res = [i,j]   # 寻找与(x,y)距离最小的栅格点
    return res  # 返回与目标栅格距离最小的点的序号

def getPicture(x, fpdb):  # RSS向量，栅格
    MAX_value = -0.5
    result = np.zeros((len(fpdb), len(fpdb[0])), dtype = float)  # 样本图，len(fpdb)行，len(fpdb[0])列
    No_datalist = []
    for i in range(len(fpdb)):  # 栅格每个横坐标
        for j in range(len(fpdb[0])):  # 栅格每个纵坐标
#            print('i = %d , j = %d'%(i,j))
            if len(fpdb[i][j][1]) <= 0:  # 无RSS向量
                No_datalist.append([i,j])  # 记录该点
                continue
            db_rss = fpdb[i][j][3]  # db_rss = 栅格指纹
            temp = []
            for k in range(len(x)):
                if (x[k]!=0 and db_rss[k]!=0):  # 当训练样本的RSS向量和栅格指纹的第K个AP对应的RSS不为0时
                    temp.append(math.pow(x[k]-db_rss[k],2))  # 计算平方差
            if temp == []:
                No_datalist.append([i,j])
#                result[i,j] = MAX_value
            else:
                result[i,j] = math.sqrt(sum(temp))/len(temp)  # 计算x和栅格指纹的欧氏距离
                if result[i,j]> MAX_value:  # 找到欧氏距离最大值
                    MAX_value = result[i,j]

    for loc in No_datalist:  # 对于那些没有收到RSS信号的点，直接赋值为1.2*欧氏距离最大值
        result[loc[0],loc[1]] = 1.2 * MAX_value
#        print('The Max value is: %f'%(1.2 * MAX_value))
    return result  # 返回样本图

def localization_train(picture_data,real_cor,graph,K):
    res = np.zeros([len(picture_data),5])  # 样本数量 * 5的矩阵
    for i in range(len(picture_data)):  #遍历样本
        res[i,0], res[i,1] = real_cor[i][0]/100,real_cor[i][1]/100  # res[i,0]和res[i,1]表示该点的真实坐标（单位：米）
        tempMatrix = picture_data[i][0]  # 信号空间距离图
        a,b = tempMatrix.shape[0],tempMatrix.shape[1]  # 信号空间距离图的高（行）和宽（列）
        Mat1d = tempMatrix.flatten()  # 将tempMatrix降维至一维（横向，把第二行放到第一行这种），只针对numpy的数组和矩阵，这是一个欧氏距离一维矩阵
        Mat1d_order= np.sort(Mat1d)[:K]  # 排序后，取mat1d[0],到mat1d[K-1],相当于对欧氏距离进行排序，取K个最小的欧氏距离
        idxMaxid = np.zeros([len(Mat1d_order),])  # 初始化一个1 * len(mat1d,_order)的矩阵（1*K）
        row = []
        col = []
        for j in range(len(Mat1d_order)):
            # np.where(condition),满足condition则输出满足条件的坐标
            idxMaxid[j] = np.where(Mat1d == Mat1d_order[j] )[0][0]  # 输出mat1d中从小到大的k个元素的序号，存入inxMaxid内
            row.append(int(idxMaxid[j]/b))  # 取整数，即为行数
            col.append(int(idxMaxid[j]%b))  # 取余数，即为列数
            Mat1d_order[j] = 1/(Mat1d_order[j]+0.001)  # 加0.001后求倒数
        Mat1d_order = Mat1d_order/np.sum(Mat1d_order)  # 权值，WKNN
        x1 = 0; y1 = 0;
        for j in range(len(Mat1d_order)):
            x1 = x1 + Mat1d_order[j]*(graph[row[j]][col[j]][0]/100)  # 权值*第j个最小值的x坐标（米）
            y1 = y1 + Mat1d_order[j]*(graph[row[j]][col[j]][1]/100)  # 权值*第j个最小值的y坐标（米）
        
        res[i,2], res[i,3] = x1, y1  # # res[i,0]和res[i,1]表示该点的真实坐标（单位：米）;res[i,2]和res[i,3]表示该点WKNN求出的坐标
    
    res[:,4] = np.sqrt( np.power(res[:,0]-res[:,2],2) + np.power(res[:,1]-res[:,3],2) )  # 该样本的误差
    error = np.mean(res[:,4])  # 所有训练集样本的WKNN的平均误差
    return res, error

def localization(picture_data,graph,K):
    res = np.zeros([len(picture_data),5])
    for i in range(len(picture_data)):
        res[i,0], res[i,1] = picture_data[i][1][0]/100,picture_data[i][1][1]/100
        tempMatrix = picture_data[i][0]
        a,b = tempMatrix.shape[0],tempMatrix.shape[1]
        Mat1d = tempMatrix.flatten()
        Mat1d_order= np.sort(Mat1d)[:K]
        idxMaxid = np.zeros([len(Mat1d_order),])
        row = []
        col = []
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

#数据加扰动部分

#导入原始数据，扰动坐标

# 扩展数据集（supplement）
path1 = '..\\pickleData\\raw_ratio\\average_data\\room_'+str(ROOM)+'_train.pkl'
path2 = '..\\pickleData\\raw_ratio\\average_data\\room_'+str(ROOM)+'_val.pkl'
path3 = '..\\pickleData\\raw_ratio\\average_data\\room_'+str(ROOM)+'_test.pkl'
path4 = '..\\pickleData\\different_grid\\06\\room'+str(ROOM)+'_06.pkl'
path5 = '..\\pickleData\\different_grid\\10\\room'+str(ROOM)+'_10.pkl'
path6 = '..\\pickleData\\different_grid\\12\\room'+str(ROOM)+'_12.pkl'

with open(path1, 'rb') as f:
    train_data = pickle.load(f)
with open(path2, 'rb') as f:
    ratio_val = pickle.load(f)
with open(path3, 'rb') as f:
    ratio_test = pickle.load(f)
with open(path4,'rb') as f:
    graph_06=pickle.load(f)
with open(path5,'rb') as f:
    graph_10=pickle.load(f)
with open(path6,'rb') as f:
    graph_12=pickle.load(f)

temp = train_data.copy(deep=True)

for i in range(train_data.shape[0]):
    add = np.random.normal(loc=mean, scale=std, size=2)  # 创建2个正态分布，均值为0，标准差为300，
    train_data.loc[i,'xLabel'] = train_data.loc[i,'xLabel'] + add[0]  # 对坐标x附加上一个正态分布，扰动坐标
    train_data.loc[i,'yLabel'] = train_data.loc[i,'yLabel'] + add[1]  # 对坐标y附加上一个正态分布，扰动坐标

disturb_path = '..\\pickleData\\raw_ratio\\average_data\\disturbed_data\\room_'+str(ROOM)+'_disturb'+str(std)+'_train_disturbed.pkl'  # 保存扰动了坐标的训练集

with open(disturb_path,'wb') as f:
    pickle.dump(train_data,f)

diff = np.zeros( (train_data.shape[0],4), dtype=np.float )
diff[:,0] = np.array(train_data.loc[:,'xLabel'])  # 扰动后X坐标
diff[:,1] = np.array(train_data.loc[:,'yLabel'])  # 扰动后y坐标
diff[:,2] = np.array(temp.loc[:,'xLabel'])  # 原始x坐标
diff[:,3] = np.array(temp.loc[:,'yLabel'])  # 原始y坐标

error = list(np.sqrt(np.square(diff[:,0]-diff[:,2])+np.square(diff[:,1]-diff[:,3])))


#归入栅格
#---------------------------------计算训练数据属于哪个栅格，归类进去，构建栅格指纹--------------------------------------------
#0.6m
fpdb_06 = [[[[],[],[],[]] for col in range(len(graph_06[0]))] for row in range(len(graph_06))]
for i in range(len(train_data)):
    x , y = train_data.loc[i,'xLabel'],train_data.loc[i,'yLabel']  # (x,y)为扰动后的训练集坐标
    target_grid = find_the_shortest_grid(x,y,graph_06)  # 与(x,y)距离最小的栅格即为目标栅格target_grid
    fpdb_06[target_grid[0]][target_grid[1]][0] = graph_06[target_grid[0]][target_grid[1]]  # 目标栅格坐标
    fpdb_06[target_grid[0]][target_grid[1]][1].append(list(train_data.iloc[i,1:762]))  # 与目标栅格最近的点(x,y)处的RSS向量，可以是多个
    fpdb_06[target_grid[0]][target_grid[1]][2].append([x,y])  # 与目标栅格最近的点(x,y)的坐标，可以是多个，与RSS向量对应
# 上述获得的fpdb_06每个栅格点含有多个训练集指纹
# 遍历fpdb_06
for i in range(len(fpdb_06)):
    for j in range(len(fpdb_06[0])):
        if len(fpdb_06[i][j][1])>0:  #含有RSS向量（多个）
            temp_res = []
            rss = np.array(fpdb_06[i][j][1])  # 该栅格内所有RSS向量
            for k in range(rss.shape[1]):  # 遍历每一列（每个AP的RSS）
                sumt = np.sum(rss[:,k])  # 该AP的所有RSS和
                if sumt < 0:  # 说明该AP收到了RSS
                    temp_res.append(sumt/np.count_nonzero(rss[:,k]))  # 求平均作为该AP的栅格指纹
                else:
                    temp_res.append(0) # 该AP未收到RSS信号，置0
            fpdb_06[i][j][3] = temp_res  # 该栅格的栅格指纹
        else:
            fpdb_06[i][j][3] = fpdb_06[i][j][1]  # 未收到RSS信号处理方式

#1m
fpdb_10 = [[[[],[],[],[]] for col in range(len(graph_10[0]))] for row in range(len(graph_10))]
for i in range(len(train_data)):
    x , y = train_data.loc[i,'xLabel'],train_data.loc[i,'yLabel']
    target_grid = find_the_shortest_grid(x,y,graph_10)
    fpdb_10[target_grid[0]][target_grid[1]][0] = graph_10[target_grid[0]][target_grid[1]]
    fpdb_10[target_grid[0]][target_grid[1]][1].append(list(train_data.iloc[i,1:762]))
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
for i in range(len(train_data)):
    x , y = train_data.loc[i,'xLabel'],train_data.loc[i,'yLabel']
    target_grid = find_the_shortest_grid(x,y,graph_12)
    fpdb_12[target_grid[0]][target_grid[1]][0] = graph_12[target_grid[0]][target_grid[1]]
    fpdb_12[target_grid[0]][target_grid[1]][1].append(list(train_data.iloc[i,1:762]))
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

#------------------------------------构造图片数据-------------------------------------
#------------------训练集------------------------------
#训练集0.6
#K_value = 1
indexlist = list(train_data.index)
picture_data_train_06 = [[None,None] for x in range(train_data.shape[0])]  # 建立一个train_data.shape[0]*train_data.shape[0]的内容为None的列表
count = 0
real_cor = []
for i in indexlist:
    print('train data 0.6 : %d， listlen: %d'%(count,len(picture_data_train_06)))
    test_rss_ext = list(train_data.iloc[i,1:762])  # RSS向量，不含坐标
    output = getPicture(test_rss_ext, fpdb_06)  # 输入该训练样本的RSS向量，栅格，输出为len(fpdb_06)*len(fpdb_06[0])的矩阵图，每个样本值为RSS向量与栅格指纹的欧氏距离
    # count指代第count个样本，一个样本包括了样本的[[RSS信号距离图]，[扰动后训练样本的坐标]]
    picture_data_train_06[count][0],picture_data_train_06[count][1] = output, [train_data.loc[i,'xLabel'], train_data.loc[i,'yLabel']]
    real_cor.append([temp.loc[i,'xLabel'], temp.loc[i,'yLabel']])  # 训练样本未经扰动的坐标
    count += 1  # 记录样本数
# 输入为0.6*0.6的信号空间距离矩阵，训练样本未经扰动的坐标，0.6*0.6的栅格坐标，K值
[cor_train_06, error_train_06] = localization_train(picture_data_train_06,real_cor,graph_06,K_value)
# 返回cor_train_06 = res([真实x,真实y，WKNN的x，WKNN的y,误差]（单位：米）)；error_train_06 = error（WKNN的平均误差）
print('error_train_06 of train set %f  :'%error_train_06)


#训练集1.0
indexlist = list(train_data.index)
picture_data_train_10 = [[None,None] for x in range(train_data.shape[0])]
count = 0
real_cor = []
for i in indexlist:
    print('train data 1.0 : %d， listlen: %d'%(count,len(picture_data_train_10)))
    test_rss_ext = list(train_data.iloc[i,1:762])
    output = getPicture(test_rss_ext, fpdb_10)
    picture_data_train_10[count][0],picture_data_train_10[count][1] = output, [train_data.loc[i,'xLabel'], train_data.loc[i,'yLabel']]
    real_cor.append([temp.loc[i,'xLabel'], temp.loc[i,'yLabel']])
    count += 1

[cor_train_10, error_train_10] = localization_train(picture_data_train_10,real_cor,graph_10,K_value)
print('error_train_10 of train set %f  :'%error_train_10)

#训练集1.2
indexlist = list(train_data.index)
picture_data_train_12 = [[None,None] for x in range(train_data.shape[0])]
count = 0
real_cor = []
for i in indexlist:
    print('train data 1.2 : %d， listlen: %d'%(count,len(picture_data_train_12)))
    test_rss_ext = list(train_data.iloc[i,1:762])
    output = getPicture(test_rss_ext, fpdb_12)
    picture_data_train_12[count][0],picture_data_train_12[count][1] = output, [train_data.loc[i,'xLabel'], train_data.loc[i,'yLabel']]
    real_cor.append([temp.loc[i,'xLabel'], temp.loc[i,'yLabel']])
    count += 1

[cor_train_12, error_train_12] = localization_train(picture_data_train_12,real_cor,graph_12,K_value)
print('error_train_12 of train set %f  :'%error_train_12)


#------------------验证集------------------------------
#验证集0.6

indexlist = list(ratio_val.index)
picture_data_val_06 = [[None,None] for x in range(ratio_val.shape[0])]
count = 0
for i in indexlist:
    print('val data 0.6 : %d， listlen: %d'%(count,len(picture_data_val_06)))
    test_rss_ext = list(ratio_val.iloc[i,1:762])
    output = getPicture(test_rss_ext, fpdb_06)
    picture_data_val_06[count][0],picture_data_val_06[count][1] = output, [ratio_val.loc[i,'xLabel'], ratio_val.loc[i,'yLabel']]
    count += 1

[cor_val_06, error_val_06] = localization(picture_data_val_06,graph_06,K_value)
print('error_val_06 of val set %f  :'%error_val_06)


#验证集1.0
indexlist = list(ratio_val.index)
picture_data_val_10 = [[None,None] for x in range(ratio_val.shape[0])]
count = 0
for i in indexlist:
    print('val data 1.0 : %d， listlen: %d'%(count,len(picture_data_val_10)))
    test_rss_ext = list(ratio_val.iloc[i,1:762])
    output = getPicture(test_rss_ext, fpdb_10)
    picture_data_val_10[count][0],picture_data_val_10[count][1] = output, [ratio_val.loc[i,'xLabel'], ratio_val.loc[i,'yLabel']]
    count += 1

[cor_val_10, error_val_10] = localization(picture_data_val_10,graph_10,K_value)
print('error_val_10 of val set %f  :'%error_val_10)

#验证集1.2
indexlist = list(ratio_val.index)
picture_data_val_12 = [[None,None] for x in range(ratio_val.shape[0])]
count = 0
for i in indexlist:
    print('val data 1.2 : %d， listlen: %d'%(count,len(picture_data_val_12)))
    test_rss_ext = list(ratio_val.iloc[i,1:762])
    output = getPicture(test_rss_ext, fpdb_12)
    picture_data_val_12[count][0],picture_data_val_12[count][1] = output, [ratio_val.loc[i,'xLabel'], ratio_val.loc[i,'yLabel']]
    count += 1

[cor_val_12, error_val_12] = localization(picture_data_val_12,graph_12,K_value)
print('error_val_12 of val set %f  :'%error_val_12)

#------------------测试集------------------------------
#测试集0.6

indexlist = list(ratio_test.index)
picture_data_test_06 = [[None,None] for x in range(ratio_test.shape[0])]
count = 0
for i in indexlist:
    print('test data 0.6 : %d， listlen: %d'%(count,len(picture_data_test_06)))
    test_rss_ext = list(ratio_test.iloc[i,1:762])
    output = getPicture(test_rss_ext, fpdb_06)
    picture_data_test_06[count][0],picture_data_test_06[count][1] = output, [ratio_test.loc[i,'xLabel'], ratio_test.loc[i,'yLabel']]
    count += 1

[cor_test_06, error_test_06] = localization(picture_data_test_06,graph_06,K_value)
print('error_test_06 of test set %f  :'%error_test_06)


#测试集1.0
indexlist = list(ratio_test.index)
picture_data_test_10 = [[None,None] for x in range(ratio_test.shape[0])]
count = 0
for i in indexlist:
    print('test data 1.0 : %d， listlen: %d'%(count,len(picture_data_test_10)))
    test_rss_ext = list(ratio_test.iloc[i,1:762])
    output = getPicture(test_rss_ext, fpdb_10)
    picture_data_test_10[count][0],picture_data_test_10[count][1] = output, [ratio_test.loc[i,'xLabel'], ratio_test.loc[i,'yLabel']]
    count += 1

[cor_test_10, error_test_10] = localization(picture_data_test_10,graph_10,K_value)
print('error_test_10 of test set %f  :'%error_test_10)

#测试集1.2
indexlist = list(ratio_test.index)
picture_data_test_12 = [[None,None] for x in range(ratio_test.shape[0])]
count = 0
for i in indexlist:
    print('test data 1.2 : %d， listlen: %d'%(count,len(picture_data_test_12)))
    test_rss_ext = list(ratio_test.iloc[i,1:762])
    output = getPicture(test_rss_ext, fpdb_12)
    picture_data_test_12[count][0],picture_data_test_12[count][1] = output, [ratio_test.loc[i,'xLabel'], ratio_test.loc[i,'yLabel']]
    count += 1

[cor_test_12, error_test_12] = localization(picture_data_test_12,graph_12,K_value)
print('error_test_12 of test set %f  :'%error_test_12)

path11 = '..\\pickleData\\raw_ratio\\pics\\06\\disturbed\\train_room_'+ str(ROOM)+'_disturb'+str(std)+'.pkl'
path12 = '..\\pickleData\\raw_ratio\\pics\\06\\disturbed\\train_KNN_localization_room_'+ str(ROOM)+'_disturb'+str(std)+'.pkl'
path13 = '..\\pickleData\\raw_ratio\\pics\\06\\disturbed\\val_room_'+ str(ROOM)+'_disturb'+str(std)+'.pkl'
path14 = '..\\pickleData\\raw_ratio\\pics\\06\\disturbed\\val_KNN_localization_room_'+ str(ROOM)+'_disturb'+str(std)+'.pkl'
path15 = '..\\pickleData\\raw_ratio\\pics\\06\\disturbed\\test_room_'+ str(ROOM)+'_disturb'+str(std)+'.pkl'
path16 = '..\\pickleData\\raw_ratio\\pics\\06\\disturbed\\test_KNN_localization_room_'+ str(ROOM)+'_disturb'+str(std)+'.pkl'

path21 = '..\\pickleData\\raw_ratio\\pics\\10\\disturbed\\train_room_'+ str(ROOM)+'_disturb'+str(std)+'.pkl'
path22 = '..\\pickleData\\raw_ratio\\pics\\10\\disturbed\\train_KNN_localization_room_'+ str(ROOM)+'_disturb'+str(std)+'.pkl'
path23 = '..\\pickleData\\raw_ratio\\pics\\10\\disturbed\\val_room_'+ str(ROOM)+'_disturb'+str(std)+'.pkl'
path24 = '..\\pickleData\\raw_ratio\\pics\\10\\disturbed\\val_KNN_localization_room_'+ str(ROOM)+'_disturb'+str(std)+'.pkl'
path25 = '..\\pickleData\\raw_ratio\\pics\\10\\disturbed\\test_room_'+ str(ROOM)+'_disturb'+str(std)+'.pkl'
path26 = '..\\pickleData\\raw_ratio\\pics\\10\\disturbed\\test_KNN_localization_room_'+ str(ROOM)+'_disturb'+str(std)+'.pkl'

path31 = '..\\pickleData\\raw_ratio\\pics\\12\\disturbed\\train_room_'+ str(ROOM)+'_disturb'+str(std)+'.pkl'
path32 = '..\\pickleData\\raw_ratio\\pics\\12\\disturbed\\train_KNN_localization_room_'+ str(ROOM)+'_disturb'+str(std)+'.pkl'
path33 = '..\\pickleData\\raw_ratio\\pics\\12\\disturbed\\val_room_'+ str(ROOM)+'_disturb'+str(std)+'.pkl'
path34 = '..\\pickleData\\raw_ratio\\pics\\12\\disturbed\\val_KNN_localization_room_'+ str(ROOM)+'_disturb'+str(std)+'.pkl'
path35 = '..\\pickleData\\raw_ratio\\pics\\12\\disturbed\\test_room_'+ str(ROOM)+'_disturb'+str(std)+'.pkl'
path36 = '..\\pickleData\\raw_ratio\\pics\\12\\disturbed\\test_KNN_localization_room_'+ str(ROOM)+'_disturb'+str(std)+'.pkl'

with open(path11,'wb') as f:
    pickle.dump(picture_data_train_06,f)
with open(path12,'wb') as f:
    pickle.dump(cor_train_06,f)
with open(path13,'wb') as f:
    pickle.dump(picture_data_val_06,f)
with open(path14,'wb') as f:
    pickle.dump(cor_val_06,f)
with open(path15,'wb') as f:
    pickle.dump(picture_data_test_06,f)
with open(path16,'wb') as f:
    pickle.dump(cor_test_06,f)

with open(path21,'wb') as f:
    pickle.dump(picture_data_train_10,f)
with open(path22,'wb') as f:
    pickle.dump(cor_train_10,f)
with open(path23,'wb') as f:
    pickle.dump(picture_data_val_10,f)
with open(path24,'wb') as f:
    pickle.dump(cor_val_10,f)
with open(path25,'wb') as f:
    pickle.dump(picture_data_test_10,f)
with open(path26,'wb') as f:
    pickle.dump(cor_test_10,f)

with open(path31,'wb') as f:
    pickle.dump(picture_data_train_12,f)
with open(path32,'wb') as f:
    pickle.dump(cor_train_12,f)
with open(path33,'wb') as f:
    pickle.dump(picture_data_val_12,f)
with open(path34,'wb') as f:
    pickle.dump(cor_val_12,f)
with open(path35,'wb') as f:
    pickle.dump(picture_data_test_12,f)
with open(path36,'wb') as f:
    pickle.dump(cor_test_12,f)

print(np.mean(np.array(error))/100)

print('--------------------room is  %d  --------------------------'%(ROOM))
print('--------------------data set is  %d  --------------------------'%(data_set))