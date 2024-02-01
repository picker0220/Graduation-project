# 导入工具包

import numpy as np
from imutils import contours
import argparse
import imutils
import cv2
import myutils

# 设置参数
parser = argparse.ArgumentParser()  # 创建解释器<必需步骤>
parser.add_argument("-i", "--image", required=True, help="path to input image")
parser.add_argument("-t", "--template", required=True, help="path to template OCR-A image")
args = vars(parser.parse_args())
# print(args)

# 指定信用卡类型
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}


# cv_show函数定义
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# 读取模板处理
img = cv2.imread(args["template"])  # 读取
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度处理
ref = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)[1]
refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
# print(np.array(refCnts).shape)
refCnts = myutils.sort_contours(refCnts, method="left-to-right")[0] # [0]表示使用该函数第一个返回值
digits = {}

# 遍历所有轮廓
for (i, c) in enumerate(refCnts):
    # print(i, c)
    # 计算外接矩形并且resize到合适大小
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[ y:y + h, x:x + w] # range of interests 先按行像素选，再按列像素选
    # print(roi.shape)
    roi = cv2.resize(roi, (57, 88))
    # cv_show('',roi)

    digits[i] = roi
# for i in digits:
#     cv_show('', digits[i])


# 初始化卷积核
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


# 预处理输入图像
image = cv2.imread(args["image"])
# cv_show('', image)
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv_show('', gray)

# 礼帽操作突出更明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
# cv_show('',tophat)

gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)  # 计算x方向梯度
gradY = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)  # 计算x方向梯度
cv2.convertScaleAbs(gradX)
cv2.convertScaleAbs(gradY)
gradXY = cv2.addWeighted(gradX, 0.5, gradY, 0.5, 0)
gradXY = cv2.convertScaleAbs(gradXY)
# cv_show('', gradXY)

# 利用闭操作，先膨胀再腐蚀，将数字连在一起
gradXY = cv2.morphologyEx(gradXY, cv2.MORPH_CLOSE, rectKernel)
# cv_show('',gradXY)
thresh = cv2.threshold(gradXY, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# cv_show('',thresh)

# 计算轮廓
threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = image.copy()
cv2.drawContours(cnt, threshCnts, -1, (0, 0, 255), 3)
# cv_show('', cnt)
locs = []

# 遍历所有轮廓
for (i, c) in enumerate(threshCnts):
    # print(i, c)
    (x, y, w, h) = cv2.boundingRect(c)
    prop = w / float(h)
    if prop > 2.5 and prop < 4.0:

        if (w > 40 and w < 55) and (h > 10 and h < 20):

            locs.append((x, y, w, h))

# 将筛选出的轮廓按顺序排序
locs = sorted(locs, key=lambda x:x[0])
output = []

# 遍历每一个轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    groupOutput = []
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    # cv_show('group', group)
    # 预处理
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # cv_show('group',group)
    # 计算每一组的轮廓
    digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

    # 计算每一组中的每一个数值
    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y+h, x:x+w]
        roi = cv2.resize(roi, (57, 88))
        # cv_show('', roi)

        scores = []

        # 计算每一个模板匹配的得分
        for (digit, digitROI) in digits.items():
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF_NORMED)
            # 返回值是minval,maxval,minloc,maxloc,所以只关心第二个返回值。
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        # 得到匹配度最高的数字
        groupOutput.append(str(np.argmax(scores)))

    # 画出来
    cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # 得到结果
    output.extend(groupOutput)

# 打印结果
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)