import cv2
import numpy as np


def showcam():
    img_file = "lenna.png"

    while True:

        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
        thresh_np = np.zeros_like(img)  # 원본과 동일한 크기의 0으로 채워진 이미지
        for x in range(128,193):
            thresh_np[img == x] = 255  # 127보다 큰 값만 255로 변경
        print(thresh_np)
        cv2.imshow('gray2', img)
        cv2.imshow('thr', thresh_np)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.release()
    cv2.destroyAllWindows()

showcam()