import cv2
import numpy as np
from Tools.scripts.fixcid import rep


def my_masking(img, my_Mask):
    if len(img.shape) == 3:
        b, g, r = cv2.split(img)
        return cv2.merge((my_masking(b, my_Mask), my_masking(g, my_Mask), my_masking(r, my_Mask)))
    elif len(img.shape) == 2:
        xx, yy = img.shape
        pad_size = np.int(np.floor(my_Mask.shape[0] / 2))
        new_img = np.float64(np.zeros(img.shape))
        scan_img = np.float64(rep(img, pad_size))  # number of pad size is here!
        for ii in range(pad_size, xx + pad_size):
            for jj in range(pad_size, yy + pad_size):
                mask_res = np.sum(scan_img[ii - pad_size: ii + pad_size + 1,
                                  jj - pad_size:jj + pad_size + 1] * my_Mask)
                new_img[ii - pad_size, jj - pad_size] = mask_res
        new_img = np.abs(new_img)
        new_img -= np.min(new_img)
        new_img *= (255 / np.max(new_img))
        # new_img[new_img > 255] = 255
        # new_img[new_img < 0] = 0

        return np.uint8(new_img)
        #return np.float64(new_img)
    else:
        return None

org_img = cv2.imread("../report2/lenna.png")
Gaussian_Filter_Mask = (1/273.0) * np.array([[1, 4, 7, 4, 1],
                                             [4, 16, 26, 16, 4],
                                             [7, 26, 41, 26, 7],
                                             [4, 16, 26, 16, 4],
                                             [1, 4, 7, 4, 1]], dtype=np.float64)
Sobel_Gradient_Mask_X = np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]], dtype=np.float64)
Sobel_Gradient_Mask_Y = np.array([[1, 2, 1],
                                  [0, 0, 0],
                                  [-1, -2, -1]], dtype=np.float64)

aa = cv2.imread("lenna.png")
bb = my_masking(aa,Sobel_Gradient_Mask_X)



cv2.imshow('img2d', bb.astype(np.unit8))
cv2.watikey()
cv2.destroyAllWindows()
