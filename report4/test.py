
import cv2
import numpy as np
import random


def add_salt_and_pepper_noise(image, percent_noise):
    if (percent_noise > 50):
        return -1
    global pixel_count

    while (True):
        for x in range(0, image.shape[0]):
            for y in range(0, image.shape[1]):

                mode = random.randrange(0, 50)

                if (mode == 1):
                    # 소금픽셀 추가
                    image[x, y] = 255
                    pixel_count = pixel_count + 1

                if (mode == 9):
                    # 후추 픽셀 추가
                    image[x, y] = 0
                    pixel_count = pixel_count + 1

                # noise_percent = float(pixel_count*100)/(image.shape[0] * image.shape[1])

                if pixel_count > ((image.shape[0] * image.shape[1]) * percent_noise / 100):

                    return image




pixel_count = 0

img = cv2.imread("lenna.png",cv2.IMREAD_GRAYSCALE)
salt = add_salt_and_pepper_noise(img,20)
width,height = img.shape
members = [(0,0)] * 5
members2= [(0,0)] * 5
save = [(0)] * 3

newimg = np.zeros_like(img)
newimg2 = np.zeros_like(img)
for i in range(1,width-1):
    for j in range(1,height-1): # 대각선 방향 및 십자가 방향의 값들을 모아서 정렬한다.

        # 대각선
        members[0] = salt[i-1,j-1]
        members[1] = salt[i, j]
        members[2] = salt[i + 1, j + 1]
        members[3] = salt[i - 1, j + 1]
        members[4] = salt[i + 1, j - 1]
        members.sort()
        save[0] = members[2]  # 중간값을 구해서 넣어준다.

        #십자가
        members2[0] = salt[i-1,j]
        members2[1] = salt[i,j-1]
        members2[2] = salt[i, j]
        members2[3] = salt[i,j+1]
        members2[4] = salt[i+1,j]
        members2.sort()
        save[1] = members2[2]  # 중간값을 구해서 넣어준다.
        save[2] = members[1]



        save.sort
        newimg[i, j] = save[1]







cv2.imshow("hybrid_median",newimg)
cv2.imshow("salt_and_pepper_noise",salt)

cv2.waitKey(0)
cv2.destroyAllWindows()




