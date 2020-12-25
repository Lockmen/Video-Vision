import cv2
import numpy
import numpy as np


def add_gaussian_noise(img_src, std):


    #노이즈가 발생한 이미지를 저장할 변수 생성(오버플로우를 방지하기 위해 float64형 사용)
    img_noisy = numpy.zeros((len(img_src),len(img_src[0])))
    for h in range(len(img_src)):
        for w in range(len(img_src[0])):
            #평균0 표준편차가 1인 정규분포를 가지는 난수 발생
            std_norm = numpy.random.normal()
            #인자로 받은 표준편차와 곱
            random_noise = std*std_norm
            #원본 값에 발생한 난수에 따른 노이즈를 합
            img_noisy[h,w] = img_src[h,w]+random_noise
    #노이즈가 발생한 이미지를 반환
    return img_noisy




image = cv2.imread('lenna.png')
image2 = add_gaussian_noise(image,64)

rows, cols = image.shape[:2]

low_pass_filter_3x3 = np.ones((3, 3), np.float32) / 9.0
low_pass_filter_5x5 = np.ones((5, 5), np.float32) / 25.0

cv2.imshow('Source', image)

dst = cv2.filter2D(image, -1, low_pass_filter_3x3)
cv2.imshow('low_pass_filter_3x3', dst)

dst = cv2.filter2D(image2, -1, low_pass_filter_5x5)
cv2.imshow('low_pass_filter_5x5', dst)





cv2.waitKey()