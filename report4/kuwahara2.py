import cv2
import numpy as np
import cv2 as cv
import time
import math


def kuwahara_normal_filter(image, kernel=7):
    pad_size = kernel // 2

    height, width, channel = image.shape[0], image.shape[1],0
    out_image = image.copy()

    pad_image = np.zeros((height + pad_size * 2, width + pad_size * 2, channel))
    for c in range(channel):
        pad_image[:, :, c] = np.pad(out_image[:, :, c], [pad_size, pad_size], channel)

    for h in range(height):
        for w in range(width):
            for c in range(channel):
                # identify the area 1,2,3,4 range
                # in pad image
                cur_point_index = (h + pad_size, w + pad_size, c)
                area = np.zeros((4, kernel // 2 + 1, kernel // 2 + 1))

                area[0] = pad_image[h:(cur_point_index[0] + 1), w:(cur_point_index[1] + 1), c]
                area[1] = pad_image[h:(cur_point_index[0] + 1), cur_point_index[1]:(cur_point_index[1] + pad_size + 1),
                          c]
                area[2] = pad_image[cur_point_index[0]:(cur_point_index[0] + 1 + pad_size), w:(cur_point_index[1] + 1),
                          c]
                area[3] = pad_image[cur_point_index[0]:(cur_point_index[0] + 1 + pad_size),
                          cur_point_index[1]:(cur_point_index[1] + 1 + pad_size), c]

                std_area = [np.std(area[0]), np.std(area[1]), np.std(area[2]), np.std(area[3])]
                min_std_area_index = np.argwhere(std_area == np.min(std_area))[0, 0]

                out_image[h, w, c] = np.sum(area[min_std_area_index]) / (len(area[min_std_area_index]) ** 2)
    return out_image


def cal_normal_integral_image(image):
    height, width, channel = image.shape[0], image.shape[1], image.shape[2]
    integral_image = np.zeros((height + 1, width + 1, channel))

    for c in range(channel):
        integral_image[:, :, c] = np.vstack((np.zeros(width + 1), np.hstack((np.zeros((height, 1)), image[:, :, c]))))

    for h in range(height):
        for w in range(width):
            for c in range(channel):
                integral_image[h + 1, w + 1, c] = integral_image[h + 1, w, c] + integral_image[h, w + 1, c] - \
                                                  integral_image[h, w, c] + integral_image[h + 1, w + 1, c]
    return integral_image


def cal_fast_integral_image(image):
    height, width, channel = image.shape[0], image.shape[1], image.shape[2]
    integral_image = np.zeros((height + 1, width + 1, channel))

    for c in range(channel):
        integral_image[:, :, c] = np.vstack((np.zeros(width + 1), np.hstack((np.zeros((height, 1)), image[:, :, c]))))

    for h in range(height):
        sum_v = [0, 0, 0]
        for w in range(width):
            for c in range(channel):
                sum_v[c] += integral_image[h + 1, w + 1, c]
                integral_image[h + 1, w + 1, c] = integral_image[h, w + 1, c] + sum_v[c]
    return integral_image


src = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
kuwa = kuwahara_normal_filter(src)

# inte = cal_fast_integral_image(src)

cv2.imshow('kuwa', kuwa.astype(np.uint8))
# cv2.imshow('inte', inte)
# print(cv2.__version__)


cv2.waitKey()
cv2.destroyAllWindows()
