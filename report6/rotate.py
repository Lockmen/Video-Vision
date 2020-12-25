import numpy as np
import cv2
import math

angle = 30
radians = float(angle * (math.pi / 180))
img = cv2.imread('Lenna_color.png', 0)
width, height = img.shape
scale =2
maxx = int(math.sqrt(width ** 2 + height ** 2))
maxy = maxx
a = int(width / 2)
b = int(height / 2)
rotate_img = np.zeros((maxx, maxy), dtype="uint8")
move_img = np.zeros((maxx, maxy), dtype="uint8")
move_img2 = np.zeros((maxx, maxy), dtype="uint8")
zoom_in = np.zeros((maxx, maxy), dtype="uint8")
zoom_out = np.zeros((maxx, maxy), dtype="uint8")

for i in range(0, maxx):
    for j in range(0, maxy):
        x = (i - a) * math.cos(radians) + (j - b) * math.sin(radians) + a
        y = -(i - a) * math.sin(radians) + (j - b) * math.cos(radians) + b

        if x < width and y < height and x > 0 and y > 0:
            rotate_img[i, j] = img[int(x), int(y)]


#move
for i in range(75, maxx):
    for j in range(25, maxy):
        if i < width and j < height and i > 0 and j > 0:
            a = img[i - 75,j - 25]
            move_img2[i,j] =  a



#zoom in
for i in range(0, maxx):
    for j in range(0, maxy):

        b= img[(int)(i/scale),(int)(j/scale)]
        zoom_in[i,j] = b

#zoom out

d=int(width / 2)
f =int(height / 2)

for i in range(0,  d):
    for j in range(0, f):
        c= img[(int)(i*scale),(int)(j*scale)]
        zoom_out[i,j] = c










cv2.imwrite("rotated" + str(angle) + ".png", rotate_img)
cv2.imshow("move",move_img2)
cv2.imshow("zoom_in",zoom_in)
cv2.imshow("zoom_out",zoom_out)

cv2.waitKey()
cv2.destroyAllWindows()