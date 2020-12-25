import cv2
import numpy
import numpy as np

def padding(img,  masksize):
    row, col = img.shape[0],img.shape[1]
    sum = np.zeros((row,col))

    for y in range (row):
        for x in range(col):

            for i in range(-1* masksize //2, masksize//2):
                for j in range(-1 * masksize//2, masksize//2):
                    new_y = y + i;
                    new_x = x + j

                    if new_y < 0: new_y = 0
                    elif new_y > row - 1: new_y = row -1

                    if new_x < 0: new_x = 0
                    elif new_x > col - 1: new_x = col -1

    save = sum[y,x]


    return save






def get_gaussian(size=3, sigma=1): # 3x3 마스크 사용(2차원)
    #sigma 값을 조절해서 필터의 weight를 조절한다. 시그마가 클수록 영상은 더 부드러워진다.

    center = (int)(size/2)
    kernel = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            diff = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            kernel[i, j] = np.exp(-(diff ** 2) / 2 * sigma ** 2)
    return kernel/np.sum(kernel)




def my_filtering(img,mask):
    '''
    :param img: Gaussian filtering을 적용 할 이미지
    :param kernel: 이미지에 적용 할 Gaussian Kernel
    :param boundary: 경계 처리에 대한 parameter (0 : zero-padding, default, 1: repetition, 2:mirroring)
    :return: 입력된 Kernel로 gaussian filtering이 된 이미지.g_input.cols - 1) new_x = img_
    '''
    # 이미지 행열
    row = len(img)
    col = len(img[0])
    mask_y, mask_x = mask.shape[0], mask.shape[1]
    pad_image = padding(img,3) # 패딩처리
    filtered_img = np.zeros((row, col), dtype=np.float32)  # 음수 소수점 처리위해 float형

# filtering 부분
    for i in range(row):
        for j in range(col):
            a = np.multiply(mask, pad_image[i:i + mask_y, j:j + mask_x])
            filtered_img[i, j] = np.sum(a)  # filter * image

    return filtered_img

def my_filtering2(img, kernel):
    '''
    :param img: Gaussian filtering을 적용 할 이미지
    :param kernel: 이미지에 적용 할 Gaussian Kernel
    :param boundary: 경계 처리에 대한 parameter (0 : zero-padding, default, 1: repetition, 2:mirroring)
    :return: 입력된 Kernel로 gaussian filtering이 된 이미지.
    '''
    # 이미지 행열
    row, col = len(img), len(img[0])
    # 커널 행열, arr.shape : array의 shape를 나타낸다
    ksizeY, ksizeX = kernel.shape[0], kernel.shape[1]

    pad_image = padding(img, (ksizeY, ksizeX)) # 패딩처리
    filtered_img = np.zeros((row, col), dtype=np.float32)  # 음수 소수점 처리위해 float형
    # filtering 부분
    for i in range(row):
        for j in range(col):
            a=np.multiply(kernel, pad_image[i:i + ksizeY, j:j + ksizeX])
            filtered_img[i, j] = np.sum(a)  # filter * image

    return filtered_img

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


img1 = cv2.imread("lenna.png")
noise =

Blurring_filter_3x3 = np.ones((3, 3), np.float32) / 9.0
gaus2d = get_gaussian()
img2d = my_filtering(img1,Blurring_filter_3x3)

cv2.imshow('img2d', img2d.astype(np.unit8))
cv2.watikey()
cv2.destroyAllWindows()



