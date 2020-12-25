

import numpy
import cv2
from PIL import Image
import numpy as np
import random
def add_gaussian_noise(img_src, std):


    #노이즈가 발생한 이미지를 저장할 변수 생성(오버플로우를 방지하기 위해 float64형 사용)
    img_noisy = numpy.zeros((len(img_src),len(img_src[0])))
    for h in range(len(img_src)):
        for w in range(len(img_src[0])):
            #평균0 표준편차가 1인 정규분포를 가지는 난수 발생
            std_norm = numpy.random.normal()
            #인자로 받은 표준편차와 곱
            random_noise = std*std_norm
            #원본 값에 발생한 난수에 따른 노이즈를 합
            img_noisy[h,w] = img_src[h,w]+random_noise
    #노이즈가 발생한 이미지를 반환
    return img_noisy


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




def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = numpy.zeros((len(data),len(data[0])))
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final


def main():
    pixel_count = 0
    img = Image.open("lenna.png").convert(
        "L")
    arr = numpy.array(img)
    arr1=add_gaussian_noise(arr,64)
    salt = add_salt_and_pepper_noise(img,20)


    removed_noise = median_filter(arr1, 3)
    #hybrid = hybrid_median_filter(arr1,3)


    img = Image.fromarray(removed_noise)
    img2 = Image.fromarray(arr1)
    img3 = Image.fromarray(salt)



    img.show()
    img2.show()
    img3.show()



main()