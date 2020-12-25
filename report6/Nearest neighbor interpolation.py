from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math



def NN_interpolation(img, dstH, dstW):
    scrH, scrW, _ = img.shape
    retimg = np.zeros((dstH, dstW, 3), dtype=np.uint8)
    for i in range(dstH - 1): # 확대한 이미지의 크기 만큼 반복
        for j in range(dstW - 1):
            scrx = round(i * (scrH / dstH)) # 정수값으로 변환
            scry = round(j * (scrW / dstW))
            retimg[i, j] = img[scrx, scry]
    return retimg


# 계단현상 발생
im_path = 'Lenna_color.png'
image = np.array(Image.open(im_path))
image1 = NN_interpolation(image, image.shape[0] * 2, image.shape[1] * 2) # x,y 길이 두배로
image1 = Image.fromarray(image1.astype('uint8')).convert('RGB')
#image1.save('out.png')

plt.subplot(131),plt.imshow(im_path, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(image1, cmap = 'gray')
plt.show()
