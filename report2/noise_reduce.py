from random import random

import cv2
import numpy
import numpy as np
import copy

def salt_and_pepper(image, p):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - p
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random()
            if rdn < p: output[i][j] = 0
            elif rdn > thres: output[i][j] = 255
            else:
                output[i][j] = image[i][j]

    return output


img = cv2.imread('lenna.png',cv2.IMREAD_GRAYSCALE)


mean = 0
var = 10
sigma = var ** 0.5
gaussian = np.random.normal(mean, sigma, (224, 224)) #  np.zeros((224, 224), np.float32)

noisy_image = np.zeros(img.shape, np.float32)

if len(img.shape) == 2:
    noisy_image = img + gaussian
else:
    noisy_image[:, :, 0] = img[:, :, 0] + gaussian
    noisy_image[:, :, 1] = img[:, :, 1] + gaussian
    noisy_image[:, :, 2] = img[:, :, 2] + gaussian

cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
noisy_image = noisy_image.astype(np.uint8)


cv2.waitKey(0)




k = np.array([[1, 2, 1],
              [2, 4, 2],
              [1, 2, 1]])

g = k = np.array([[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]) / 9.0







img2 =  salt_and_pepper(img,0.005)



blur3 = cv2.GaussianBlur(img2, (9,9), 0)

blur = cv2.filter2D(img, -1, k)




cv2.imshow('orig', img)
cv2.imshow('blur', blur)
cv2.imshow('blur2', noisy_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

