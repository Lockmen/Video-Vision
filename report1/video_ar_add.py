import cv2
import numpy as np

alpha = 0.5 # 합성 알파값

img_file="diff_img1.jpg"
img_file2 ="diff_img3.jpg"

img1 = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img_file2,cv2.IMREAD_GRAYSCALE)


cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)



# 수식을 직접 연산해서 알파 블렌딩 작용
blended = img1 * alpha + img2 * (1-alpha)
blended = blended.astype(np.uint8) # 소수점 발생을 제거하기 위함

sub = img1-img2 # 같은 부분은 0이 되므로 검게 된다.
thresh_np = np.zeros_like(sub)  # 원본과 동일한 크기의 0으로 채워진 이미지
thresh_np[sub>0] = 255 # wrap방식(뺄셈연산으로 차이나는 부분이 0에 가까운 경우 0으로 두는 경우가 잇어서 wrap방식으로 처리 0은 그대로 두고 255(현 프로그램에서는 하지 않아도 차이를 알수 있음) )








# 차 영상을 극대화하기 위해 스레시 홀드 및 컬러로 전환

















cv2.imshow('img1 * alpha + img2 * (1-alpha)', blended)
cv2.imshow('img1',img1)
cv2.imshow('img2',img2)

cv2.imshow('sub',thresh_np)


cv2. waitKey(0)
cv2.destroyAllWindows()