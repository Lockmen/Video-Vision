from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math



def NN_interpolation(img, rate):
    scrH, scrW, _ = img.shape
    retimg = np.zeros((scrH, scrW, 3), dtype=np.uint8)
    for i in range(scrH - 1): # 확대한 이미지의 크기 만큼 반복
        for j in range(scrW - 1):
            px= (int)(j/rate)
            py= (int)(i/rate)

            #네 점으로부터의 거리비
            fx1 = (float)(j/rate) - (float)(px)
            fx2 = 1 - fx1
            fy1 = (float)(i/rate) - (float)(py)
            fy2 = 1 - fy1

            #
            w1 = fx2 * fy2
            w2= fx1 * fy2
            w3= fx2 * fy1
            w4= fx1 * fy1

            #인접한 4개의 픽셀을 찾는다.
            P1 = img[py,px]
            P2 = img[py, px + 1]
            P3 = img[py + 1, px]
            P4 = img[py + 1, px + 1]

            # 선형보간법 적용
            retimg[i,j] = w1*P1 + w2*P2 + w3*P3 + w4*P4

    return retimg


# 계단현상 발생
im_path = 'Lenna_color.png'
image = np.array(Image.open(im_path))
image1 = NN_interpolation(image, 2) # x,y 길이 두배로
image1 = Image.fromarray(image1.astype('uint8')).convert('RGB')
image1.save('out.png')