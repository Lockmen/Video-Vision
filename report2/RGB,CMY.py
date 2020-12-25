import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg


img=mpimg.imread('diff_img1.jpg',0)

rgb_scale = 255
cmyk_scale = 100


def rgb_to_cmyk(rgb, percent_gray=100):


    cmy = 1 - rgb / 255.0  # 공식 적용
    k = np.min(cmy, axis=2) * (percent_gray / 100.0)
    k[np.where(np.sum(rgb,axis=2)==0)] = 1.0
    k_mat = np.stack([k,k,k], axis=2)

    with np.errstate(divide='ignore', invalid='ignore'):
        cmy = (cmy - k_mat) / (1.0 - k_mat)
        cmy[~np.isfinite(cmy)] = 0.0

    return np.dstack((cmy, k))





def rgb_to_ycbcr(im):
    cbcr = np.empty_like(im)
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]

    cbcr[:,:,0] = .299 * r + .587 * g + .114 * b # YBCR 공식적용

    cbcr[:,:,1] = 128 - .169 * r - .331 * g + .5 * b

    cbcr[:,:,2] = 128 + .5 * r - .419 * g - .081 * b
    return np.uint8(cbcr)


plt.title('RGB image')
plt.imshow(img)
plt.show()

plt.title('YCBCR image')
img1=rgb_to_ycbcr(img)
plt.imshow(img1)

plt.show()




plt.title('CMYK image')
img2=rgb_to_cmyk(img)
plt.imshow(img2)
plt.show()