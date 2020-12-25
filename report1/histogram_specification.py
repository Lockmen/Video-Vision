#원본 영상의 누적 히스토그램 계산
#지정된 히스토그램의 누적 히스토그램 계산
#두 누적 히스토그램 간의 차이에 대한 절대값 계산
#누적 히스토그램 차이를 기반으로 회색 수준 맵 설정

import cv2
import numpy as np
from PIL import Image

_MAX_HISTO_ = 256

#히스토그램 계산 함수

def calc_histo(img_src):

    #히스토그램을 누적할 변수 생성

    histo_info = np.zeros(_MAX_HISTO_) # 0으로 채워진 256열 생성

    src_height = img_src.shape[0]

    src_width = img_src.shape[1]

    #히스토그램 계산

    for h in range(src_height):

        for w in range(src_width):

            histo_info[img_src[h,w]] += 1 # 밝기별 픽셀수 늘려가며, 총수를 저장


    return histo_info

#히스토그램 정보를 받아서 히스토그램 이미지를 반환하는 함수
#이 히스토그램을 이용하여 누적 히스토그램을 생성하고 정규화
def draw_histo(histo_info):

    #히스토그램을 그릴 공간 생성

    img_histo = np.zeros([256, _MAX_HISTO_], dtype=np.uint8) # 256/256 배열 정수형으로 생성

    histo_height = img_histo.shape[0]

    max_histo = max(histo_info)

    #히스토그램 그리기(축의 높이를 조절하기 위해 최대값 사용)

    for i in range(_MAX_HISTO_):

        cv2.line(img_histo, (i, histo_height), (i, int(histo_height-histo_info[i]/max_histo*histo_height)), 255, 1)

    return img_histo

#원본 영상을 받아 히스토그램 평활화를 수행한 결과 영상을 반환하는 함수

def histo_eq(img_src):

    #히스토그램 계산

    histo_info = calc_histo(img_src)

    #누적분포에 따라 적용할 색상정보 룩업 테이블을 위한 변수 생성

    lookup_table = np.zeros(_MAX_HISTO_)
    save =  np.zeros(_MAX_HISTO_, dtype=np.uint8)


    pixel_num = np.sum(histo_info) # 누적히스토그램 ?

    accum_val = 0

    for i in range(_MAX_HISTO_):

        accum_val += histo_info[i]

        lookup_table[i] = int(accum_val/pixel_num*255)
        save[i] = int(accum_val / pixel_num * 255)

    two = 255
    one = 254

    while one>=1:
        for i in range(one,two):

            lookup_table[i] = two

            two = one
            one =-1








    #평활화가 적용된 결과를 저장할 변수 생성

    img_he = np.zeros(img_src.shape, dtype=np.uint8)

    for h in range(img_src.shape[0]):

        for w in range(img_src.shape[1]):

            img_he[h,w] = lookup_table[img_src[h,w]] #색상정보 룩업 테이블을 이용하여 해당 위치 픽셀 색상 변경

    return img_he

#현재 실행되고 있는 경로 값을 얻어서 이미지 경로를 조합







img_src_path = "C:\\Users\\jim60\\Desktop\\Video Vision\\lenna.png"
img_src = cv2.imread(img_src_path)

#히스토그램 평활화 적용된 이미지 계산

img_he = histo_eq(img_src)

#평활화가 적용된 이미지로부터 히스토그램 정보 계산 및 히스토그램 그리기

histo_info_he = calc_histo(img_he)

img_histo_he = draw_histo(histo_info_he)





#그레이스케일로 이미지 읽기

img_src = cv2.imread(img_src_path, cv2.IMREAD_GRAYSCALE)

#히스토그램 계산

histo_info_src = calc_histo(img_src)

#히스토그램 그리기

img_histo_src = draw_histo(histo_info_src)

#원본 이미지와 히스토그램 상태 출력

cv2.imshow('src', img_src)

cv2.imshow('histo_src', img_histo_src)

cv2.imshow('he', img_he)

cv2.imshow('histo_he', img_histo_he)



cv2.waitKey(0)

cv2.destroyAllWindows()

