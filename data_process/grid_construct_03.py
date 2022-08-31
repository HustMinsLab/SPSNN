import pickle

# 栅格的构建，将对应样本的坐标都归入一个栅格中
# 构造了0.6，1.0，1.2（单位：米）的栅格
# 房间号
ROOM = 4

path1 = '..\\pickleData\\ss\\ave\\train_room_'+ str(ROOM)+'.pkl'
path2 = '..\\pickleData\\test\\ave\\test_room_'+ str(ROOM)+'.pkl'

with open(path1,'rb') as f:
    room_train = pickle.load(f)
with open(path2,'rb') as f:
    room_test = pickle.load(f)

#--------------------------对于房间区域与走廊6----------------------------------------
# interval  == 0.6:
x_label = list(set(list(room_train.loc[:,'xLabel'])))
y_label = list(set(list(room_train.loc[:,'yLabel'])))
x_label.sort()
y_label.sort()
graph_06 = [[None for col in range(len(y_label))] for row in range(len(x_label))]
for i in range(len(x_label)):
    for j in range(len(y_label)):
        graph_06[i][j] = [x_label[i],y_label[j]]

#interval  == 1:
x_label = list(set(list(room_test.loc[:,'xLabel'])))
y_label = list(set(list(room_test.loc[:,'yLabel'])))
x_label.sort()
y_label.sort()
graph_10 = [[None for col in range(len(y_label))] for row in range(len(x_label))]
for i in range(len(x_label)):
    for j in range(len(y_label)):
        graph_10[i][j] = [x_label[i],y_label[j]]

#interval  == 1.2:
x_temp = list(set(list(room_train.loc[:,'xLabel'])))
y_temp = list(set(list(room_train.loc[:,'yLabel'])))
x_temp.sort()
y_temp.sort()
x_label = x_temp[1:len(x_temp):2]
y_label = y_temp[1:len(x_temp):2]
graph_12 = [[None for col in range(len(y_label))] for row in range(len(x_label))]
for i in range(len(x_label)):
    for j in range(len(y_label)):
        graph_12[i][j] = [x_label[i],y_label[j]]

path4 = '..\\pickleData\\different_grid\\06\\room'+str(ROOM)+'_06.pkl'
path5 = '..\\pickleData\\different_grid\\10\\room'+str(ROOM)+'_10.pkl'
path6 = '..\\pickleData\\different_grid\\12\\room'+str(ROOM)+'_12.pkl'

with open(path4,'wb') as f:
    pickle.dump(graph_06,f)
with open(path5,'wb') as f:
    pickle.dump(graph_10,f)
with open(path6,'wb') as f:
    pickle.dump(graph_12,f)
