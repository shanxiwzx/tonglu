# 表格内容识别
识别方法：
1.二值化表格；
2.通过腐蚀和膨胀检测表格中的横线和竖线；
3.依照横线对倾斜表格进行旋转矫正；
4.叠加横线和竖线定位表格角点，对角点进行排序，画出完整表格；
5.依据表格角度坐标切割出要识别的内容区域；
6.采用tesseract-orc（4.0以上版本）进行识别。
