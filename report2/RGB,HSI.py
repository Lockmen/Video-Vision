from PIL import Image
import math

 #HIS 컬러 모델은 색상(Hue), 채도(Saturation), 명도(Intensity)로 구성된다.
 #h,i,s 3개영상 나오게 해야한다.
def rgbtohsi(R, G, B):
    r = R / 255   # RGB 색상의 정규화
    g = G / 255
    b = B / 255
    num = 0.5 * ((r - g) + (r - b))
    den = ((r - g) * (r - g) + (r - b) * (g - b)) ** (0.5)

    if (b <= g):

        if den != 0:
            h = math.acos(num / (den))  # h 값
        else:
            h = 0

    #HSI컬러 모델은 원뿔 모양의 좌표계로 표현된다. 색상은 원뿔 둘레를 따라 0도에서 360도의 범위를 가진 각도로 표현된다.
    # 0도는 빨강색, 120도는 초록색, 240도는 파랑색을 나타낸다.
    # 채도는 0에서 1까지의 값을 가지며 원뿔 중심으로부터의 수평거리로 표현된다.

    elif (b > g):

        if den != 0:
            h = (2 * math.pi) - math.acos(num / den)  # h 값 360-H
        else:
            h = 0
    s = 1 - (3 * min(r, g, b) / (r + g + b))  # s 값
    i = (r + g + b) / 3  # i 값

    return int(h * 180 / math.pi), int(s * 100), int(i * 255)


image = Image.open("Lenna_color.png").convert("RGB")
image_pix = image.load()
w = image.size[0]
hg = image.size[1]
for i in range(w):
    for j in range(hg):
        r, g, b = image.getpixel((i, j))
        h, s, v = rgbtohsi(r, g, b)
        image_pix[i, j] = (h, s, v)
image.save("hsi_new.jpg")
image.show()