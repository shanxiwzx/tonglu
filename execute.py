# -*- coding:utf-8 -*-
"""
@author  wangzhaoxi
@date  20190521
@enterprise GDPI
@待改进   缩放后再旋转，保证图片不超出边框
"""
import cv2
import numpy as np
import pytesseract
import math,sys
import re
import json

split_alpha = 'abcdefghijklmnopqrstuvwxyz$!~`=WMI‘§{[*&#@|:%l°_'
class TableLocation:
    """
    目标：对表格中的订单编号进行识别
    步骤：1.读取表格，进行二值化处理；
          2.通过腐蚀和膨胀操作识别表格的横线和竖线；
          3.依照横线对倾斜表格进行旋转矫正
          4.叠加横线和竖线定位表格角点，对角点进行排序；
          5.画出完整表格；
          6.在原图中切割出订单编号格子；
          7.对订单编号格子进行识别。
    """
    def __init__(self, path):
        self.path = path
        self.img = cv2.imread(path)

    def read_table(self, min, max):                                                                #二值化图片
        self.img1 = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)                                     # 图片灰度化
        img1 = cv2.GaussianBlur(self.img1, (1, 1), 0)                                              # 高斯模糊处理，消除噪音
        ret, self.binary = cv2.threshold(img1, min, max, cv2.THRESH_BINARY_INV)
        return self.binary

    def erode_dilate(self, img, scale1, scale2, iterations):                                       #腐蚀膨胀操作
        rows, cols = img.shape
        #提取横线
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale1, 1))
        erosion = cv2.erode(img, kernel, iterations=iterations)
        dilation = cv2.dilate(erosion, kernel, iterations=iterations)
        #提取竖线
        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale2))
        erosion1 = cv2.erode(img, kernel1, iterations=iterations)
        dilation1 = cv2.dilate(erosion1, kernel1, iterations=1)
        return dilation, dilation1

    def rotate(self, dilation, img):                                      #图像旋转
        rows, cols = dilation.shape
        lines = cv2.HoughLinesP(dilation, 1, np.pi/180, 100, 100, 10)
        line = lines[0][0]                                                #获取检测到的第一条直线（任何一条都可以）
        tanA = math.atan((line[3]-line[1])/(line[2]-line[0]))             #计算tan值
        angle = tanA*(180/np.pi)                                          #计算倾斜角
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)           #旋转模板
        self.JImg = cv2.warpAffine(img, M, (2*cols, 2*rows))              #进行旋转
        return self.JImg

    def locate_table(self, dilation, dilation1):
        cross_dot = cv2.bitwise_and(dilation, dilation1)                  # 求横竖线条的交叉点
        merge = cv2.add(dilation, dilation1)                              #无实际作用，可视化查看
        # 识别黑白图中的白色点，即横竖线叠加后的表格角点
        # cv2.imshow("wzx", merge)
        # cv2.waitKey(0)
        ys, xs = np.where(cross_dot > 0)
        def new_dot(s,threshold):                                         #对横线上的点和竖线上的进行排序，两点之间距离小于threshold则剔除
            newlist = []
            xs1 = np.sort(s)
            for i in range(len(xs1) - 1):
                if (xs1[i + 1] - xs1[i] > threshold):
                    newlist.append(xs1[i])
            newlist.append(xs1[i])
            return newlist                                                #获取有效的角点的横坐标
        self.newlist_xs = new_dot(xs, 50)
        newlist_ys = new_dot(ys, 10)
        if newlist_ys[0] < 30: del newlist_ys[0]
        return self.newlist_xs, newlist_ys, merge

    def image_split(self, mylistx, mylisty, img):                                    #截取指定的方格
        ROI1 = img[mylisty[3]:mylisty[4], mylistx[1]:mylistx[2]-2]                   #模板一的格子
        # cv2.imshow("wzx", ROI1)
        # cv2.waitKey(0)
        self.text2 = pytesseract.image_to_string(ROI1).strip(split_alpha)     # 读取文字，此为默认英文
        if "." in self.text2:                                                             #个别识别错误的位置调整
            ROI2 = img[mylisty[2]:mylisty[3], mylistx[1]:mylistx[2]]
            self.text2 = pytesseract.image_to_string(ROI2).strip(split_alpha)
        elif len(self.newlist_xs) < 5 and bool(re.search(r'\d', self.text2)) == False:                                               #模板二的格子
            ROI2 = img[mylisty[0]:mylisty[1], mylistx[0]:mylistx[1]]
            # cv2.imshow("wzx", ROI2)
            # cv2.waitKey(0)
            self.text2 = pytesseract.image_to_string(ROI2).strip(split_alpha)
        elif bool(re.search(r'\d', self.text2)) == False or len(self.text2) < 10:    #若识别结果中无数字或者长度小于10，再进行适当调整
            ROI1 = img[mylisty[3]+2:mylisty[4]-12, mylistx[1]+3:mylistx[2]-5]
            self.text2 = pytesseract.image_to_string(ROI1).strip(split_alpha)
            if bool(re.search(r'\d', self.text2)) == False:
                ROI1 = img[mylisty[3]-5:mylisty[4], mylistx[2] + 5:mylistx[3] - 5]
                self.text2 = pytesseract.image_to_string(ROI1).strip(split_alpha)
                if bool(re.search(r'\d', self.text2)) == False:
                    ROI1 = img[mylisty[5] + 2:mylisty[6], mylistx[1] + 3:mylistx[2] - 5]
                    # cv2.imshow("wzx", ROI1)
                    # cv2.waitKey(0)
                    self.text2 = pytesseract.image_to_string(ROI1).strip(split_alpha)
            elif "." in self.text2 or len(self.text2) < 9:
                ROI1 = img[mylisty[2]:mylisty[3], mylistx[1]:mylistx[2] - 2]
                self.text2 = pytesseract.image_to_string(ROI1).strip(split_alpha)
            else:
                self.text2 = self.text2
        else:
            self.text2 = self.text2
        return self.text2

if "__main__" == __name__:
    if len(sys.argv) > 1:                                 # 读取参数名作为文件名
        path = sys.argv[1]
    else:
        path = "./0523/0523/2.jpg"
    tableP = TableLocation(path)
    binary = tableP.read_table(190, 255)
    dilation, _ = tableP.erode_dilate(binary, 120, 30, 1)
    img = tableP.rotate(dilation, tableP.img1)
    img1 = tableP.rotate(dilation, tableP.binary)
    dilation, dilation1 = tableP.erode_dilate(img1, 40, 20, 1)
    lx, ly, merge = tableP.locate_table(dilation, dilation1)
    #print(lx, ly, len(ly))
    text = tableP.image_split(lx, ly, img)               #在矫正后的原图上裁剪进行识别
    text = re.sub("[^A-Z0-9()/: ]", '', text)
    bItem = {}
    bItem["data"] = text
    jsonArr = json.dumps(bItem, ensure_ascii=False)
    print(jsonArr)
