import torch
import torch.backends.cudnn as cudnn
import cv2
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import datetime  
import matplotlib.pyplot as plt
import glob
from CNN_model import  *
from CNN_Data_inout import *

labelnum=11
root="./data/"
savepath="./"
imgpath = "./data/images/"


trainSet =LoadPartDataset(txt=root+'train.txt', root=root)
test_data=LoadPartDataset(txt=root+'test.txt', root=root)
test_infor=loadTestData(txt=root+'test.txt')
train_loader = DataLoader(dataset=trainSet, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)

device="cuda"
model = Net(labelnum)
model.cuda()
cudnn.benchmark = True
epochnum=50
loss_acc_mat= np.zeros((epochnum,5),dtype=np.float32)#epoch,train_los, train_acc, eval_loss,val_acc
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam (model.parameters(), lr=0.0001)
model.train()

Train_loss_list = []
Train_acc_list = []
Test_loss_list = []
Test_acc_list = []

for epoch in range(epochnum):
    print('epoch {}'.format(epoch + 1))
    loss_acc_mat[epoch,0]=epoch

    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for trainData, trainLabel in train_loader:
        trainData, trainLabel = Variable(trainData.cuda()), Variable(trainLabel.cuda())
        optimizer.zero_grad()
        out = model(trainData)
        loss = loss_func(out, trainLabel)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == trainLabel).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    now_time = datetime.datetime.now()
    now_time=datetime.datetime.strftime(now_time,'%Y-%m-%d %H:%M:%S')
    Train_loss6f = float(train_loss / len(trainSet))
    Train_acc6f =  float(train_acc / len(trainSet))
    print(now_time,'Train Loss: %.6f, Acc: %.6f'%(Train_loss6f, Train_acc6f))
    Train_loss_list.append(Train_loss6f)
    Train_acc_list.append(Train_acc6f)
    loss_acc_mat[epoch,1]=    train_loss/(len(trainSet))
    loss_acc_mat[epoch,2]=    train_acc/(len(trainSet))

    # evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    y_test_com =None# np.zeros((len_test,1),dtype=np.float32)
    y_predicted_com =None #np.zeros((len_test,1),dtype=np.float32)
    pos_test=0
    kk=0
    for testData,testLabel in test_loader:
         testData, testLabel = Variable(testData.cuda()), Variable(testLabel.cuda())
         out = model( testData)
         loss = loss_func(out, testLabel )
         eval_loss += loss.item()
         pred = torch.max(out, 1)[1]
         num_correct = (pred == testLabel).sum()
         eval_acc += num_correct.item()
         y_predicted= pred.cpu() .detach().numpy()
         len1=len( testLabel)
         y_test= testLabel.reshape((len1,1))
         y_predicted=y_predicted.reshape((len1,1))
         y_test= y_test.cpu(). detach().numpy()
         if kk==0:
             y_predicted_com =y_predicted
             y_test_com=y_test
         else:
             y_predicted_com =np.vstack((y_predicted_com ,y_predicted))
             y_test_com=  np.vstack((y_test_com ,y_test))
         kk +=1
    loss_acc_mat[epoch,3]=    eval_loss / (len(test_data))
    loss_acc_mat[epoch,4]=    eval_acc  / (len(test_data))
    np_data_full=np.hstack((y_test_com ,y_predicted_com ))
    model.zero_grad()
    now_time = datetime.datetime.now()
    now_time=datetime.datetime.strftime(now_time,'%Y-%m-%d %H:%M:%S')
    acc=int((eval_acc /  len(test_data))*10000)
    Test_loss6f = float(eval_loss / len(test_data))
    Test_acc6f = float(eval_acc / len(test_data))
    print(now_time, 'Test Loss: %.6f, Acc: %.6f' % (Test_loss6f, Test_acc6f))
    Test_loss_list.append(Test_loss6f)
    Test_acc_list.append(Test_acc6f)

    len1=len(y_test_com)
    pred_ct = np.zeros((labelnum,4),dtype=np.float32)
    pred_cross = np.zeros((labelnum,labelnum),dtype=np.float32)
    tt=0

    for i in range(2):
        pred_ct[i,0] =tt
        tt+=1
    for i in range(len1):
        label=int(y_test_com[i])
        pred=int(y_predicted_com[i])
        pred_cross [label,pred] = pred_cross[label,pred]+1
        if pred==label:
            pred_ct[label,1] =pred_ct[label,1]+1

        else:
            pred_ct[label,2] =pred_ct[label,2]+1

    if pred_ct[:,1].all() == 0 and pred_ct[:,2].all() == 0:
        pred_ct[:,3] = 0
    else:
        pred_ct[:,3] = pred_ct[:,1]/(pred_ct[:,1]+pred_ct[:,2])

    model.train()
    model.zero_grad()

epoch = range(epochnum)
# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 绘制损失值图像
plt.figure()
plt.plot(epoch,Train_loss_list,label='训练集损失值')
plt.plot(epoch,Test_loss_list,label='测试集损失值')
plt.xticks([0,epochnum/2,epochnum])
plt.title('损失值曲线')
plt.xlabel('训练次数')
plt.ylabel('损失值')
plt.legend()
# 绘制准确率图像
plt.figure()
plt.plot(epoch,Train_acc_list,label='训练集准确率')
plt.plot(epoch,Test_acc_list,label='测试集准确率')
plt.xticks([0,epochnum/2,epochnum])
plt.title('准确度曲线')
plt.xlabel('训练次数')
plt.ylabel('准确度')
plt.legend()
plt.show()
# 保存模型
torch.save(model.state_dict(), savepath + 'model_val.pth')
