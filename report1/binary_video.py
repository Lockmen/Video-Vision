import cv2
import numpy as np


def showcam(): # 영상의 이진화
    try:
        print ('open cam')
        cap = cv2.VideoCapture(0)
    except:
        print ('Not working')
        return
    cap.set(3, 500)
    cap.set(4, 500)
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 영상을 그레이스케일로 설정
        thresh_np = np.zeros_like(gray)  # 원본과 동일한 크기의 0으로 채워진 이미지
        thresh_np[gray > 127] = 255  # 127보다 큰 값만 255로 변경
        print(thresh_np)
        cv2.imshow('frame', frame)
        cv2.imshow('gray2', gray)
        cv2.imshow('thr', thresh_np)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
showcam()