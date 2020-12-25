import cv2
import numpy as np


src = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
hom = np.zeros_like(src) # 이미지 크기와 똑같지만 0으로 채워진 배열 생성
dif = hom.copy()

for y in range(512):
    for x in range(512):
        arr = []
        for i in [-1, 0, 1]:
            try:
                a = int(abs(src[y-1][x-i]))
                b = int(abs(src[y+1][x+i]))
                arr.append(abs(a - b))
            except IndexError:
                continue
        try:
            a = int(abs(src[y][x - 1]))
            b = int(abs(src[y][x + 1]))
            arr.append(abs(a - b))
        except IndexError:
            continue
        dif[y][x] = max(arr)

res = np.hstack([src, dif])
cv2.imshow('res', res)
cv2.waitKey(0)
