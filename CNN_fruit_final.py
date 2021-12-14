import torch.backends.cudnn as cudnn
import cv2
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
from CNN_model import  *
from  CNN_Data_inout import *

labelnum = 11
root="./data/"
savepath="./"
device = "cuda"
softmax = nn.Softmax(dim=-1)


def classfy(imgpath):
    model = Net(labelnum)
    model.cuda()
    cudnn.benchmark = True

    ckpt = torch.load(savepath + "model_val.pth", map_location=device)  # Load your best model
    model.load_state_dict(ckpt)

    test_data = Loadimage(imgpath)
    test_loader = DataLoader(dataset=test_data, batch_size=64)

    y_predicted_com = None  # np.zeros((len_test,1),dtype=np.float32)
    kk = 0


    for testData in test_loader:
        testData = Variable(testData.cuda())
        out = model(testData)

        probs = softmax(out)

        score_list, class_list = probs.max(dim=-1)

        pred = torch.max(out, 1)[1]
        score_lists = score_list.cpu().detach().numpy()
        y_predicted = pred.cpu().detach().numpy()
        len1 = len(testData)
        y_predicted = y_predicted.reshape((len1, 1))
        score_lists = score_lists.reshape((len1, 1))
        if kk == 0:
            y_predicted_com = y_predicted
            score_lists_com = score_lists
        else:
            y_predicted_com = np.vstack((y_predicted_com, y_predicted))
            score_lists_com = np.vstack((score_lists_com, score_lists))
        kk += 1

    np_data_full = y_predicted_com
    score_lists = score_lists_com
    np_data_full = [int(x) for x in np_data_full]
    score_lists = [float(x) for x in score_lists]
    pred_infor = infer_from_prob(np_data_full, root + 'labelnum.txt')


    return pred_infor, score_lists

def opencvLoad(imgPath, resizeH, resizeW):
    image = cv2.imread(imgPath)
    image = cv2.resize(image, (resizeH, resizeW), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 1, 0))
    image = torch.from_numpy(image)
    return image

if __name__== "__main__" :

    imgpath=["D:/Computer Vision/application/code/Interfruit/apple_1.jpg"]
    for i in range(1):
        pred_infor, score_lists=classfy(imgpath)
        print(pred_infor[0])
        print(score_lists[0])