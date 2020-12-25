import numpy as np, cv2
import matplotlib.pylab as plt

img1 = np.zeros((200,400), dtype= np.uint8)
img2 = np.zeros((200,400), dtype= np.uint8)
img3 = np.zeros((200,400), dtype= np.uint8)
img4 = np.zeros((200,400), dtype= np.uint8)
img5 = np.zeros((200,400), dtype= np.uint8)
img1[:,:200] = 255 # 왼쪽은 검은색(0), 오른쪽은 흰색(255)
img2[100:200, :] = 255 # 위쪽은 검은색(0), 오른쪽은 흰색(255)

src_height = img1.shape[0]

src_width = img2.shape[1]

# 히스토그램 계산

for h in range(src_height):

    for w in range(src_width):
        if img1[h,w]==255 == img2[h,w] == 255:
            img3[h,w] = 255
        else:
            img3[h,w]=0

        if img1[h, w] == 255 or img2[h, w] == 255:
            img4[h, w] = 255

        if (img1[h,w] == img2[h,w]):
            img5[h, w] = 0
        else:
            img5[h,w] = 255








cv2.imshow('img1', img1)
cv2.imshow('img2',img2)
cv2.imshow('and', img3)
cv2.imshow('or',img4)
cv2.imshow('xor',img5)


cv2. waitKey(0)
cv2.destroyAllWindows()







