import cv2
import numpy as np

image = cv2.imread('lenna.png',cv2.IMREAD_GRAYSCALE)

sharpening_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpening_2 = np.array([[-1, -1, -1, -1, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, 2, 9, 2, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, -1, -1, -1, -1]]) / 9.0

cv2.imshow("original",image)
dst = cv2.filter2D(image, -1, sharpening_1)
cv2.imshow('Sharpening1', dst)

dst = cv2.filter2D(image, -1, sharpening_2)
cv2.imshow('Sharpening2', dst)
cv2.waitKey()