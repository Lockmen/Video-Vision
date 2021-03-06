import numpy as np
import cv2
img = cv2.imread("lenna.png",cv2.IMREAD_GRAYSCALE)
row ,col = img.shape
def binaryconvt(img) :
    list = []
    for i in range(row):
        for j in range(col):
             list.append (np.binary_repr( img[i][j] ,width=8  ) ) # 2진수 출력
    return list


im1 = binaryconvt(img)
def bitplane(bitimgval , img1D ):
    bitList = [int(i[bitimgval])for i in img1D]
    return bitList

bit8 = np.array( bitplane(0,im1 ) ) * 128
bit7 = np.array( bitplane(1,im1) ) * 64

bit6 = np.array( bitplane(2,im1 ) ) * 32
bit5 = np.array( bitplane(3,im1) ) * 16

bit4 = np.array( bitplane(4,im1) ) * 4

combine = bit8 + bit7
comb = np.reshape(combine,(row,col))
cv2.imwrite("comb(8+7).jpeg",comb)

combine2 = bit8 + bit7 + bit6 + bit5
comb2 = np.reshape(combine2,(row,col))
cv2.imwrite("comb(all).jpeg",comb2)

bit8 = np.reshape(bit8,(row,col))
cv2.imwrite("8bit.jpg" , bit8 )
bit7 = np.reshape(bit7,(row,col))
cv2.imwrite("7bit.jpg",bit7)

bit6 = np.reshape(bit6,(row,col))
cv2.imwrite("6bit.jpg",bit6)
bit5 = np.reshape(bit5,(row,col))
cv2.imwrite("5bit.jpg",bit5)

bit4 = np.reshape(bit4,(row,col))
cv2.imwrite("4bit.jpg",bit4)


cv2.waitKey(0)
cv2.destroyAllWindows()