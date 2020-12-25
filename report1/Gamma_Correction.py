import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(img, c=1, r=1.2): # g(x,y) = (f(x,y)/255)^1/r )* 255 적용
    out = img.copy()
    out /= 255. # out = out/255
    out = (1/c * out) ** (1/r)

    out *=255 + 0.5
    out = out.astype(np.uint8) # 픽셀 데이터가 최대 255이므로 부호없는 8비트 unint8로 한다.

    return out
#Read image
img = cv2.imread("lenna.png").astype(np.float)

#Gamma correction
out = gamma_correction(img)

#save reulst
cv2.imshow("result",out)
cv2.waitKey(0)
cv2.imwrite("out.jpg",out)





