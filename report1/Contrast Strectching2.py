import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

gray_img = Image.open("lenna.png").convert("LA")#밝기와 알파값을 이용해서 Grayscale로 변환
gray_img.show()#grayscale로 변환된 흑백 이미지를 출력
row = gray_img.size[0] # row 이미지 크기 알아냄
col = gray_img.size[1] # col 이미지 크기 알아냄
stretch_img = Image.new("L", (row, col))#새 흑백이미지를 생성.
high = 0
low = 255

for x in range(1 , row):
    for y in range(1, col):
        if high < gray_img.getpixel((x,y))[0] : # 입력영상 최대값 찾기
            high = gray_img.getpixel((x,y))[0]
        if low > gray_img.getpixel((x,y))[0]: # 입력영상 최소값 찾기
            low = gray_img.getpixel((x,y))[0]
for x in range(1 , row):
    for y in range(1, col):
        stretch_img.putpixel((x,y), int((gray_img.getpixel((x,y))[0]-low)*255/(high-low)))
stretch_img.show()#스트레칭된 이미지 출력

y = gray_img.histogram()
y = y[0:256]
x = np.arange(len(y))
plt.title("original hist")
plt.bar(x, y)
plt.show()

y = stretch_img.histogram()
x = np.arange(len(y))
plt.title("stretch hist")
plt.bar(x, y)
plt.show()