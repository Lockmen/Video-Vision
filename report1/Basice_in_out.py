import cv2

img_file = "C:\\Users\\LIM JONG HYEON'\\Desktop\\Video_Vision1\\animenz.jpg" # 표시할 이미지 경로
save_file = "C:\\Users\\LIM JONG HYEON'\\Desktop\\Video_Vision1\\animenz_gray.jpg"

img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE) # 이미지를 읽어서 img 변수에 할당
cv2.imshow('IMG',img) # 읽은 이미지를 화면에 표시
cv2.imwrite(save_file,img) # 파일로 저장, 포맷은 확장자에 따름
cv2.waitKey() # 키가 입력될 때까지 대기
cv2.destroyAllWindows() # 창 모두 닫기