import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt

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



img = cv2.imread('lenna.png',cv2.IMREAD_GRAYSCALE )
kernel = np.ones((5,5), np.float32) / 25

a= add_gaussian_noise(img,64)
dst = cv2.filter2D(a, -1, kernel)



plt.subplot(121),plt.imshow(a),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()