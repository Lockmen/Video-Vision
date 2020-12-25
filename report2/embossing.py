import cv2
import numpy as np


def emboss(image):
    kernel = np.array([[0,-1,-1],
                            [1,0,-1],
                            [1,1,0]])
    return cv2.filter2D(image, -1, kernel)




img = cv2.imread("lenna.png",cv2.IMREAD_GRAYSCALE)
embo = emboss(img)
cv2.imshow("original",img)
cv2.imshow("embossing",embo)
cv2.waitKey(0)


