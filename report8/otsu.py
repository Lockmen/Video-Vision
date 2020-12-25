import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 원본 이미지 확인
img = np.array(Image.open('lenna.png').convert('L'))
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()

histo = []


def otsu(arg):
    # 입력 이미지의 픽셀수
    N = len(arg) * len(arg[0])
    # 히스토그램
    histogram = [0] * 256
    for line in arg:
        for pixel in line:
            histogram[pixel] = histogram[pixel] + 1

    threshold = 0
    sumTotal = 0.0
    sumB = 0.0
    w0 = 0
    w1 = 0
    varMax = 0.0
    # 전체 weighted sum calculation
    for i in range(len(histogram)):
        sumTotal += i * histogram[i]

    for i in range(len(histogram)):
        w0 += histogram[i]
        w1 = N - w0
        if w0 == 0:
            continue
        if w1 == 0:
            break
        # sigma_(i=1 to k) i*p_i
        sumB += float(i * histogram[i])
        # calc left m1, right m2 - mean
        m1 = sumB / w0
        m2 = (sumTotal - sumB) / w1
        # between-class variance
        varBetween = float(w0) * float(w1) * (m1 - m2) * (m1 - m2)
        if varBetween > varMax:
            varMax = varBetween
            threshold = i
    return threshold


thres = otsu(img)
for i in range(len(img)):
    for j in range(len(img[i])):
        if thres > img[i][j]:
            img[i][j] = 0
        else:
            img[i][j] = 255

plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()