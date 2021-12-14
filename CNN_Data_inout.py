import cv2
from torch.utils.data import Dataset, DataLoader
import numpy as np
from CNN_model import  *

def saveauc(savefilename, cvnum, i, kk, auc):
    target = open(savefilename, 'a')

    s1 = str(cvnum) + "\t" + str(i) + "\t" + str(kk) + "\t" + str(auc) + "\n"
    target.write(s1)
    target.close()


def savelabelenumdata(savefilename, probdata):
    target = open(savefilename, 'a')
    len1 = np.shape(probdata)[0]
    for i in range(len1):
        label = int(probdata[i, 0])
        pos = "%.0f" % probdata[i, 1]
        neg = "%.0f" % probdata[i, 2]
        acc = "%.4f" % probdata[i, 3]
        s = str(label) + "\t" + str(pos) + "\t" + str(neg) + "\t" + str(acc) + "\n"

        target.write(s)
    target.close()


def savecrosspreddata(savefilename, crossdata, labelnum):
    target = open(savefilename, 'a')
    len1 = labelnum
    len2 = labelnum
    for i in range(len1):
        s = ""

        for j in range(len2):
            s += str("%.0f" % crossdata[i, j]) + "\t"
        s += "\n"
        target.write(s)
    target.close()


def saveprobdata(savefilename, probdata, test_infor):
    target = open(savefilename, 'a')

    len1 = len(probdata[:, 0])

    for i in range(len1):
        s = str("%.0f" % probdata[i, 0]) + "\t" + str("%.0f" % probdata[i, 1]) + "\t" + test_infor[i]
        s += "\n"
        target.write(s)
    target.close()


def savelossaccdata(savefilename, lossdata):
    target = open(savefilename, 'a')

    len1 = len(lossdata[:, 0])
    s = "epoch\ttrainloss\trainacc\ttestloss\ttestacc\n"
    target.write(s)

    for i in range(len1):
        s = str("%.0f" % lossdata[i, 0]) + "\t" + str("%.7f" % lossdata[i, 1]) + "\t" + str(
            "%.7f" % lossdata[i, 2]) + "\t" + str("%.7f" % lossdata[i, 3]) + "\t" + str("%.7f" % lossdata[i, 4])
        s += "\n"
        target.write(s)
    target.close()


def saveauchead(savefilename):
    target = open(savefilename, 'a')
    s0 = "cvnum\trepeatnum\tepcohnum\tauc\n"
    target.write(s0)

    target.close()


def saveteststr(savefilename, s):
    target = open(savefilename, 'a')

    target.write(s)

    target.close()


# -----------------ready the dataset--------------------------
def opencvLoad(imgPath, resizeH, resizeW):
    image = cv2.imread(imgPath)
    image = cv2.resize(image, (resizeH, resizeW), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 1, 0))
    image = torch.from_numpy(image)
    return image

class Loadimage(Dataset):
    def __init__(self, imagelist):
        self.imgs = imagelist

    def __getitem__(self, item):
        image = self.imgs[item]
        img = opencvLoad(image, 227, 227)
        return img

    def __len__(self):
        return len(self.imgs)


class LoadPartDataset(Dataset):
    def __init__(self, txt, root):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()

            labelList = int(words[0])
            imageList = root + words[1]
            imgs.append((imageList, labelList))
        self.imgs = imgs

    def __getitem__(self, item):
        image, label = self.imgs[item]
        # print(image)
        img = opencvLoad(image, 227, 227)
        return img, label

    def __len__(self):
        return len(self.imgs)


def loadTrainData(txt=None, root=None):
    fh = open(txt, 'r')
    imgs = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        label = int(words[0])
        image = cv2.imread(root + words[1])
        image = cv2.resize(image, (227, 227), interpolation=cv2.INTER_CUBIC)
        image = image.astype(np.float32)
        image = np.transpose(image, (2, 1, 0))
        image = torch.from_numpy(image)
        imgs.append((image, label))
    fh.close()
    return imgs


def loadTestData(txt=None):
    fh = open(txt, 'r')
    imgs = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()

        imageList = words[1]
        imgs.append(imageList)
    fh.close()
    return imgs

def infer_from_prob(prob_data, labelnum):
    fh = open(labelnum, 'r')
    classlist = []
    outtxt = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()

        classlist.append(words[1])
    del classlist[0]
    for i in prob_data:
        outtxt.append(classlist[i])
    return outtxt