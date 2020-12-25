import cv2
import numpy as np
from random import random
import copy

# 1. 마스크 내 화소값들을 크기 순으로 정렬
# 2. 알파 값에 따라 정렬된 자료들의 양쪽을 자르고
# 3. 나머지 가운데 값들을 평균한 값을 출력한다.
# 4. 알파가 최소일 때는 평균값 필터, 알파가 최대일 때는 미디언 필터처럼 행동한다.

def ftAlphaTrimmedMean(image,alpha = 0.5):
    img = image.copy()

    height, width = image.shape
    height = height - 1
    width = width - 1

    a = 0
    for i in range(1, height):
        for j in range(1, width):
            arr = []
            a = image[i - 1][j - 1] # [0,0]
            arr.append(a)

            a = image[i - 1][j]  # [0,1]
            arr.append(a)

            a = image[i - 1][j + 1]  # [0,2]
            arr.append(a)

            a = image[i][j - 1]  # [1,0]
            arr.append(a)

            a = image[i][j]  # [1,1]
            arr.append(a)

            a = image[i][j + 1]  # [1,2]
            arr.append(a)

            a = image[i + 1][j - 1]  # [2,0]
            arr.append(a)

            a = image[i + 1][j]  # [2,1]
            arr.append(a)

            a = image[i + 1][j + 1]  # [2,2]
            arr.append(a)

            arr = QuickSort(arr)
            leng = 9



            tengah = int(alpha * leng)

            for k in range(tengah,leng-tengah):
                sum = 0
                sum += arr[k]


            a = sum / (leng-2*tengah) # 나머지 가운데 값들을 평균한 값

            img[i][j] = a

    return img

# def ftAlphaTrimmedMean(image):
#     img = image.copy()
#     height, width = image.shape
#     height = height - 1
#     width = width - 1
#
#     a = 0
#     for i in range(1, height):
#         for j in range(1, width):
#             arr = []
#             a = image[i - 1][j - 1]
#             arr.append(a)
#
#             a = image[i - 1][j]
#             arr.append(a)
#
#             a = image[i - 1][j + 1]
#             arr.append(a)
#
#             a = image[i][j - 1]
#             arr.append(a)
#
#             a = image[i][j]
#             arr.append(a)
#
#             a = image[i][j + 1]
#             arr.append(a)
#
#             a = image[i + 1][j - 1]
#             arr.append(a)
#
#             a = image[i + 1][j]
#             arr.append(a)
#
#             a = image[i + 1][j + 1]
#             arr.append(a)
#             arr = QuickSort(arr)
#             leng = len(arr) - 1
#
#             tengah = int(leng / 2) 양쪽으로 자른다.
#             a = 0
#             a += arr[tengah - 2]
#             a += arr[tengah - 1]
#             a += arr[tengah]
#             a += arr[tengah + 1]
#             a += arr[tengah + 2]
#
#             a = int(a / 5)
#
#             img[i][j] = a
#
#     return img


def QuickSort(array):
    less = []
    equal = []
    greater = []

    if len(array) > 1:
        pivot = array[0]
        for x in array:
            if x < pivot:
                less.append(x)
            elif x == pivot:
                equal.append(x)
            elif x > pivot:
                greater.append(x)

        return QuickSort(less)+equal+QuickSort(greater)
    else:
        return array

def salt_and_pepper(image, p):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - p
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random()
            if rdn < p:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output





src = cv2.imread('lenna.png',cv2.IMREAD_GRAYSCALE )
noise = salt_and_pepper(src,0.01)
alpha_filter = ftAlphaTrimmedMean(noise)

cv2.imshow('original',src)
cv2.imshow('alpha_trimmmed mean filter', alpha_filter.astype(np.uint8))
cv2.imshow("s&p", noise)
cv2.waitKey()
cv2.destroyAllWindows()