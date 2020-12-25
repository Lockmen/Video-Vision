import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

start = time.time()
#이미지 로드
image_source = plt.imread('lenna.png').astype(float)
img_float32 = np.float32(image_source)


plt.figure()
plt.imshow(image_source, plt.cm.gray)
plt.xticks([]), plt.yticks([])
plt.title('Original image')


dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)

dft_shift = np.fft.fftshift(dft) # 푸리에 변환의 결과를 중심으로 이동시킴

rows, cols = image_source.shape
crow, ccol = rows/2 , cols/2  # 센터

keep_fraction = 60




f_ishift = np.fft.ifftshift(dft_shift) # 역변환


img_back = cv2.idft(f_ishift)


img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

end = time.time()
print("Time:", end-start)

fig = plt.figure(figsize=(20,10))
plt.imshow(img_back, plt.cm.gray)
plt.xticks([]), plt.yticks([])
plt.title('Reconstructed Image DFT')
plt.show()
fig.savefig('baboon_gauss60_DFT60.png')


