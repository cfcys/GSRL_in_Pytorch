import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import torch.nn.init as init

colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255],
                   [176, 48, 96], [46, 139, 87], [160, 32, 240], [255, 127, 80], [127, 255, 212],
                   [218, 112, 214], [160, 82, 45], [127, 255, 0], [216, 191, 216], [128, 0, 0], [0, 128, 0],
                   [0, 0, 128]])

class ResNet99(nn.Module):
    def __init__(self, band1, band2, imx, ncla1, l=1):
        super(ResNet99, self).__init__()
        self.l = l
        self.conv0x = nn.Conv2d(band2, 32, kernel_size=(3, 3), padding='valid')
        self.conv0 = nn.Conv2d(band1, 32, kernel_size=(3, 3), padding='valid')
        self.bn11 = nn.BatchNorm2d(64, eps=0.001, momentum=0.9)
        self.conv11 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same')
        self.conv12 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same')
        self.bn21 = nn.BatchNorm2d(64, eps=0.001, momentum=0.9)
        self.conv21 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same')
        self.conv22 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same')
        self.fc1 = nn.Linear(64, ncla1)
        # self.dconv1 = nn.ConvTranspose2d(64, 64, kernel_size=(1, 1), padding='valid')
        # self.dconv2 = nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), padding='valid')
        # self.dconv3 = nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), padding='valid')
        # self.dconv4 = nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), padding='valid')
        # f.dconv5 = nn.ConvTranspose2d(64, band1, kernel_size=(3, 3), padding='valid')
        self.dconv1 = nn.ConvTranspose2d(64, 64, kernel_size=(1, 1), padding=0)
        self.dconv2 = nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), padding=0)
        self.dconv3 = nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), padding=0)
        self.dconv4 = nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), padding=0)
        self.dconv5 = nn.ConvTranspose2d(64, band1, kernel_size=(3, 3), padding=0)
        self.bn1_de = nn.BatchNorm2d(64)
        self.bn2_de = nn.BatchNorm2d(64)

        
        init.normal_(self.conv0x.weight, mean=0.0, std=0.01)
        init.normal_(self.conv0.weight, mean=0.0, std=0.01)
        init.normal_(self.conv11.weight, mean=0.0, std=0.01)
        init.normal_(self.conv12.weight, mean=0.0, std=0.01)
        init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        
    def forward(self, input1, input2):
        # x1 = F.relu(self.bn11(self.conv0(input1)))
        # x1x = F.relu(self.bn11(self.conv0x(input2)))
        x1 = self.conv0(input1)
        # ipdb.set_trace()
        x1x = self.conv0x(input2)
        x1 = torch.cat([x1, x1x], dim=1)
        
        x11 = F.relu(self.bn11(x1))
        x11 = F.relu(self.conv11(x11))
        x11 = self.conv12(x11)
        x1 = x1 + x11
        

        if self.l == 2:
            x11 = F.relu(self.bn21(x1))
            x11 = F.relu(self.conv21(x11))
            x11 = F.relu(self.conv22(x11))
            x1 = x1 + x11

        x1 = F.adaptive_avg_pool2d(x1, (1, 1))
        x1 = torch.flatten(x1, 1)
        # ipdb.set_trace()
        # pre1 = F.softmax(self.fc1(x1),dim=1)
        pre1 = self.fc1(x1)

        x12 = x1.reshape(-1, 64, 1, 1)
        x12 = F.relu(self.bn1_de(self.dconv1(x12)))
        x12 = F.relu(self.dconv2(x12))
        x12 = F.relu(self.bn2_de(self.dconv3(x12)))
        x12 = F.relu(self.dconv4(x12))
        x12 = self.dconv5(x12)
        # print("x12.shape:",x12.shape)
        
        return pre1, x12
    
def imgDraw(label, imgName, path='./pictures', show=True):
    """
    功能：根据标签绘制RGB图
    输入：（标签数据，图片名）
    输出：RGB图
    备注：输入是2维数据，label有效范围[1,num]
    """
    row, col = label.shape
    numClass = int(label.max())
    Y_RGB = np.zeros((row, col, 3)).astype('uint8')  # 生成相同shape的零数组
    Y_RGB[np.where(label == 0)] = [0, 0, 0]  # 对背景设置为黑色
    for i in range(1, numClass + 1):  # 对有标签的位置上色
        try:
            Y_RGB[np.where(label == i)] = colors[i - 1]
        except:
            Y_RGB[np.where(label == i)] = np.random.randint(0, 256, size=3)
    plt.axis("off")  # 不显示坐标
    if show:
        plt.imshow(Y_RGB)
    os.makedirs(path, exist_ok=True)
    plt.imsave(path + '/' + str(imgName) + '.png', Y_RGB)  # 分类结果图
    return Y_RGB


def displayClassTable(n_list, matTitle=""):
    """
    功能：打印list的各元素
    输入：（list）
    输出：无
    备注：无
    """
    from pandas import DataFrame
    print("\n+--------- 原始输入数据" + matTitle + "统计结果 ------------+")
    lenth = len(n_list)  # 一共n个分类
    column = range(1, lenth + 1)
    table = {'Class': column, 'Total': [int(i) for i in n_list]}
    table_df = DataFrame(table).to_string(index=False)
    print(table_df)
    print('All available data total ' + str(int(sum(n_list))))
    print("+---------------------------------------------------+")


def listClassification(Y, matTitle=''):
    """
    功能：对标签数据计数并打印
    输入：（原始标签数据，是否打印）
    输出：分类结果
    备注：无
    """
    numClass = np.max(Y)  # 获取分类数
    listClass = []  # 用列表依次存储各类别的数量
    for i in range(numClass):
        listClass.append(len(np.where(Y == (i + 1))[0]))
    displayClassTable(listClass, matTitle)
    return listClass
