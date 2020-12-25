import cv2
import numpy as np

# 이미지 불러오기
src = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
hom = np.zeros_like(src) # 이미지 크기와 똑같지만 0으로 채워진 배열 생성

# 이미지의 크기 (512, 512) 순회
for y in range(512):
    for x in range(512):
        arr = [] # arr는 커널에서 중앙의 포인트 값을 뺀 것에 절댓값을 취한 값들이 들어감
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i is 0 and j is 0: # 자기 자신 제외
                    continue
                try:
                    point = int(src[y][x]) # 이렇게 안해주면 오버플로우가 생깁니다.
                    kernel_value = int(src[y+i][x+j])
                    arr.append(abs(point - kernel_value))
                except IndexError: # 인덱스 잘못된 건 무시
                    continue
                hom[y][x] = max(arr) # 커널 한개로부터 받은 값 (3~8개) 중 최댓값 추출

res = np.hstack([src, hom])
cv2.imshow('res', res)
cv2.waitKey(0)
