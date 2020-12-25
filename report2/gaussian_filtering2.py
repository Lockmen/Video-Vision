import cv2
import numpy
import numpy as np
import time

# 바깥쪽 패딩 채우기
def my_padding(img, shape, boundary=0):
    '''
    :param img: boundary padding을 해야할 이미지
    :param shape: kernel의 shape
    :param boundary: default = 0, zero-padding : 0, repetition : 1, mirroring : 2
    :return: padding 된 이미지.
    '''
    row, col = len(img), len(img[0])
    pad_sizeY, pad_sizeX = shape[0] // 2, shape[1] // 2
    res = np.zeros((row + (2 * pad_sizeY), col + (2 * pad_sizeX)), dtype=np.float)
    pad_row, pad_col = len(res), len(res[0])
    if pad_sizeY == 0:
        res[pad_sizeY:, pad_sizeX:-pad_sizeX] = img.copy()
    elif pad_sizeX == 0:
        res[pad_sizeY:-pad_sizeY, pad_sizeX:] = img.copy()
    else:
        res[pad_sizeY:-pad_sizeY, pad_sizeX:-pad_sizeX] = img.copy()
    if boundary == 0:
        return res
    elif boundary == 1:
        res[0:pad_sizeY, 0:pad_sizeX] = img[0, 0]  # 좌측 상단
        res[-pad_sizeY:, 0:pad_sizeX] = img[row - 1, 0]  # 좌측 하단
        res[0:pad_sizeY, -pad_sizeX:] = img[0, col - 1]  # 우측 상단
        res[-pad_sizeY:, -pad_sizeX:] = img[row - 1, col - 1]  # 우측 하단
        # axis = 1, 열반복, axis = 0, 행반복. default 0
        res[0:pad_sizeY, pad_sizeX:pad_col - pad_sizeX] = np.repeat(img[0:1, 0:], [pad_sizeY], axis=0)  # 상단
        res[pad_row - pad_sizeY:, pad_sizeX:pad_col - pad_sizeX] = np.repeat(img[row - 1:row, 0:], [pad_sizeY],
                                                                             axis=0)  # 하단
        res[pad_sizeY:pad_row - pad_sizeY, 0:pad_sizeX] = np.repeat(img[0:, 0:1], [pad_sizeX], axis=1)  # 좌측
        res[pad_sizeY:pad_row - pad_sizeY, pad_col - pad_sizeX:] = np.repeat(img[0:, col - 1:col], [pad_sizeX],
                                                                             axis=1)  # 우측
        return res
    else:
        res[0:pad_sizeY, 0:pad_sizeX] = np.flip(img[0:pad_sizeY, 0:pad_sizeX])  # 좌측 상단
        res[-pad_sizeY:, 0:pad_sizeX] = np.flip(img[-pad_sizeY:, 0:pad_sizeX])  # 좌측 하단
        res[0:pad_sizeY, -pad_sizeX:] = np.flip(img[0:pad_sizeY, -pad_sizeX:])  # 우측 상단
        res[-pad_sizeY:, -pad_sizeX:] = np.flip(img[-pad_sizeY:, -pad_sizeX:])  # 우측 하단

        res[pad_sizeY:pad_row - pad_sizeY, 0:pad_sizeX] = np.flip(img[0:, 0:pad_sizeX], 1)  # 좌측
        res[pad_sizeY:pad_row - pad_sizeY, pad_col - pad_sizeX:] = np.flip(img[0:, col - pad_sizeX:], 1)  # 우측
        res[0:pad_sizeY, pad_sizeX:pad_col - pad_sizeX] = np.flip(img[0:pad_sizeY, 0:], 0)  # 상단
        res[pad_row - pad_sizeY:, pad_sizeX:pad_col - pad_sizeX] = np.flip(img[row - pad_sizeY:, 0:], 0)  # 하단
        return res

def add_gaussian_noise(img_src,std):
    src_height = img_src.shape[0]
    src_width = img_src.shape[0]
    #노이즈가 발생한 이미지를 저장할 변수 생성(오버플로우를 방지하기 위해 float64형 사용)
    img_noisy = np.zeros(img_src.shape,dtype=np.float64)
    for h in range(src_height):
        for w in range(src_width):
            # 평균0 표준편차가 1인 정규분포를 가지는 난수 발생
            std_norm = np.random.normal()
            #인자로 받은 표준편차와 곱
            random_noise = std*std_norm
            #원본 값에 발생한 난수에 따른 노이즈를 합
            img_noisy[h,w]= img_src[h,w] + random_noise
            #노이즈가 발생한 이미지를 반환
    return img_noisy



# Gaussian kernel 생성 코드를 작성해주세요.
def my_getGKernel(shape, sigma): # 시그마가 커지면 영상이 부드러워진다.
    '''
    :param shape: 생성하고자 하는 gaussian kernel의 shape입니다. (5,5) (1,5) 형태로 입력받습니다.
    :param sigma: Gaussian 분포에 사용될 표준편차입니다. shape가 커지면 sigma도 커지는게 좋습니다.
    :return: shape 형태의 Gaussian kernel
    '''
    # a = shape[0] , b = shape[1] , (s = 2a+1, t = 2b+1)
    s = (shape[0] - 1) / 2
    t = (shape[1] - 1) / 2

    # 𝑠,𝑡 가 –a~a, -b~b의 범위를 가짐 ,  np.ogrid[-m:m+] : -m~m까지 증가하는 array를 반환한다.
    # 𝑥 :−𝑏~𝑏 범위의 Kernel에서의 x좌표(열) , 𝑦 :−𝑎~𝑎 범위의 Kernel에서의 y좌표(행)
    y, x = np.ogrid[-s:s + 1, -t:t + 1]
    # e^-(x^2 + y^2)/2𝜎^2
    # -	np.exp(x) : 𝑒^𝑥 를 구한다
    gaus_kernel = np.exp(-(x * x + y * y)) / (2. * sigma * sigma)
    # arr.sum() : array의 값을 모두 더해 반환한다.
    sum = gaus_kernel.sum()
    gaus_kernel /= sum
    return gaus_kernel


def my_filtering(img, kernel, boundary=0):
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

    pad_image = my_padding(img, (ksizeY, ksizeX), boundary=boundary) # 패딩처리
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


src = cv2.imread('lenna.png',cv2.IMREAD_GRAYSCALE )


# get Gaussian Kernal 필터 크기 홀수 x 홀수인 모든 필터를 만족해야한다.
gaus2D = my_getGKernel((51, 51), 12)
gaus1D = my_getGKernel((1, 51), 12)

sharpening_1 = np.array([[-1, -1, -1, -1, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, 2, 9, 2, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, -1, -1, -1, -1]]) / 9.0

high_sharp = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
low_sharp = np.array([[1,1,1], [1,1,1], [1,1,1]]) / 9.0


laplacian  = np.array([[0,1,0], [1,-4,1], [0,1,0]])


Blurring_filter_3x3 = np.ones((3, 3), np.float32) / 9.0
Blurring_filter_5x5 = np.ones((5, 5), np.float32) / 25.0



start = time.perf_counter()  # 시간 측정

# noise
noise = add_gaussian_noise(src,64)
noise2d = my_filtering(noise,low_sharp)



# 2D filtering
img2D = my_filtering(src, gaus2D)
end = time.perf_counter()
print("2D:", end - start)

start = time.perf_counter()
# 1D filtering
img1D = my_filtering(src, gaus1D)
img1D = my_filtering(img1D, gaus1D.T)
end = time.perf_counter()
print("1D:", end - start)


#laplacian
laplacian2d = my_filtering(src,laplacian)
end = time.perf_counter()
print("laplacian:", end - start)

#log
laplacian2d_gu = my_filtering(img2D,laplacian)
end = time.perf_counter()
print("laplacian_gu:", end - start)



#sharp
sharp2d = my_filtering(src,sharpening_1)
end = time.perf_counter()
print("sharp_lowpass:", end - start)

#low sharp
sharp2d_low = my_filtering(noise,low_sharp)
end = time.perf_counter()
print("sharp_lowpass:", end - start)

#high sharp
sharp2d_high = my_filtering(src,high_sharp)
end = time.perf_counter()
print("high_sharp:", end - start)





#Blurring
Blurring2d =  my_filtering(img2D,Blurring_filter_5x5);
end = time.perf_counter()
print("Blurring:", end - start)


#dog
Blurring2d_3x3 =  my_filtering(img2D,Blurring_filter_3x3);
dog = Blurring2d_3x3 - Blurring2d


cv2.imshow('img1D', img1D.astype(np.uint8))
cv2.imshow('img2D', img2D.astype(np.uint8))
cv2.imshow('shapr_lowpass',sharp2d_low.astype(np.uint8))
cv2.imshow('sharp_high',sharp2d_high.astype(np.uint8))
cv2.imshow('sharp',sharp2d.astype(np.uint8))
cv2.imshow('Blurring', Blurring2d.astype(np.uint8))
cv2.imshow('lapacian', laplacian2d.astype(np.uint8))
cv2.imshow('lapacian_gu', laplacian2d_gu.astype(np.uint8))
cv2.imshow('dog', dog.astype(np.uint8))
cv2.imshow('noise_reduce', noise2d.astype(np.uint8))





cv2.waitKey()
cv2.destroyAllWindows()