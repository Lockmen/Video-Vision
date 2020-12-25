import cv2
import numpy as np
from matplotlib import pyplot as plt # np.histogram(data, 도수분포구간(bin))
# cdf의 값이 0인 경우는 mask처리를 하여 계산에서 제외
# mask처리가 되면 Numpy 계산에서 제외가 됨
# 아래는 cdf array에서 값이 0인 부분을 mask처리함
img = cv2.imread("C:\\Users\\LIM JONG HYEON'\\Desktop\\Video_Vision1\\lenna.png") #bins는 도수분표구간(구분)
hist, bins = np.histogram(img.flatten(), 256, [0, 256]) # flatten은 다차원을 1차원으로 바꿈
cdf = hist.cumsum() # 누적 히스토그램
cdf_m = np.ma.masked_equal(cdf, 0) # 0인 값을 NAN으로 제거




cdf = np.ma.filled(cdf_m, 0).astype('uint8') # 히스토그램을 픽셀로 매핑
img = cdf[img]
cv2.imshow('img', img)
plt.hist(img.flatten(), 256,[0, 256], color='r')
plt.xlim([0, 256])
plt.show()