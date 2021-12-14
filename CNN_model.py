import torch

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [torch.nn.ReflectionPad2d(1),
                      torch.nn.Conv2d(in_features, in_features, 3),
                      torch.nn.InstanceNorm2d(in_features),
                      torch.nn.ReLU(inplace=True),
                      torch.nn.ReflectionPad2d(1),
                      torch.nn.Conv2d(in_features, in_features, 3),
                      torch.nn.InstanceNorm2d(in_features)]

        self.conv_block = torch.nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

    # 定义conv-bn-relu函数


def conv_relu(in_channel, out_channel, kernel, stride=1, padding=0):
    conv = torch.nn.Sequential(
        torch.nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
        torch.nn.BatchNorm2d(out_channel, eps=1e-3),
        torch.nn.ReLU(True),
    )
    return conv


# 定义incepion结构，见inception图
class inception(torch.nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5,
                 out4_1):
        super(inception, self).__init__()
        self.branch1 = conv_relu(in_channel, out1_1, 1)
        self.branch2 = torch.nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1))
        self.branch3 = torch.nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2))
        self.branch4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        output = torch.cat([b1, b2, b3, b4], dim=1)
        return output

        # -----------------create the Net and training------------------------


class Net(torch.nn.Module):
    def __init__(self, labelnum):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, 11, 4, 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(96, 256, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 384, 3, 1, 1),
            torch.nn.ReLU(),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 384, 3, 1, 1),
            torch.nn.ReLU(),
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(384, 256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 2)
        )

        self.res1 = ResidualBlock(256)

        self.inc1 = inception(256, 4, 4, 12, 1, 3, 2)  # )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(21 * 6 * 6, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, labelnum)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        dd = conv5_out.view(conv5_out.size(0), 256, 6, 6)
        dd = self.res1(dd)
        dd = self.inc1(dd)

        res = dd.view(dd.size(0), -1)
        out = self.dense(res)

        return out