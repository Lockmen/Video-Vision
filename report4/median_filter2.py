import cv2
import numpy as np
import math


def sp_noise(image, ratio):
    output = np.zeros(image.shape, np.uint8)
    ratio = ratio / 2
    thres = 1 - ratio
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < ratio:
                output[i][j][0] = 0
                output[i][j][1] = 0
                output[i][j][2] = 0
            elif rdn > thres:
                output[i][j][0] = 255
                output[i][j][1] = 255
                output[i][j][2] = 255
            else:
                output[i][j][0] = image[i][j][0]
                output[i][j][1] = image[i][j][1]
                output[i][j][2] = image[i][j][2]
    return output


def median_filter(image, ksize):
    output = np.zeros(image.shape, np.uint8)
    border = ksize // 2
    image_border = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_REFLECT_101)
    height, width = image.shape[:2]

    for k in range(3):
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                member = []

                member.append(image_border[i - 1][j - 1][k])
                member.append(image_border[i - 1][j][k])
                member.append(image_border[i - 1][j + 1][k])
                member.append(image_border[i][j - 1][k])
                member.append(image_border[i][j][k])
                member.append(image_border[i][j + 1][k])
                member.append(image_border[i + 1][j - 1][k])
                member.append(image_border[i + 1][j][k])
                member.append(image_border[i + 1][j + 1][k])

                member.sort()
                output[i - 1][j - 1][k] = member[4]

    return output


img = cv2.imread("lenna.png",cv2.IMREAD_GRAYSCALE)
a= sp_noise(img,0.2)
b= median_filter(a,3)

cv2.imshow("noise",a)
cv2.imshow("meidan",b)
cv2.waitKey(0)
