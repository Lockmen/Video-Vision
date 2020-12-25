import copy
import numpy as np
import cv2 as cv


def kuwahara_filter(image, kernel_size=5):

   
    height, width, channel = image.shape[0], image.shape[1], image.shape[2]

    r = int((kernel_size - 1) / 2)
    r = r if r >= 2 else 2

    image = np.pad(image, ((r, r), (r, r), (0, 0)), "edge")

    average, variance = cv.integral2(image)
    # 평균값
    average = (average[:-r - 1, :-r - 1] + average[r + 1:, r + 1:] -
               average[r + 1:, :-r - 1] - average[:-r - 1, r + 1:]) / (r +
                                                                 1)**2
   #분산값
    variance = ((variance[:-r - 1, :-r - 1] + variance[r + 1:, r + 1:] -
                 variance[r + 1:, :-r - 1] - variance[:-r - 1, r + 1:]) /
                (r + 1)**2 - average**2).sum(axis=2)

    def filter(i, j):
        return np.array([
            average[i, j], average[i + r, j], average[i, j + r], average[i + r,
                                                                         j + r]
        ])[(np.array([
            variance[i, j], variance[i + r, j], variance[i, j + r],
            variance[i + r, j + r]
        ]).argmin(axis=0).flatten(), j.flatten(),
            i.flatten())].reshape(width, height, channel).transpose(1, 0, 2)

    filtered_image = filter(*np.meshgrid(np.arange(height), np.arange(width)))

    filtered_image = filtered_image.astype(image.dtype)
    filtered_image = filtered_image.copy()

    return filtered_image


def main():


        frame = cv.imread("lenna.png",)
        frame2 = kuwahara_filter(frame)
        cv.imshow('original', frame)
        cv.imshow('kuwabara', frame2)
        cv.waitKey(0)

        cv.destroyAllWindows()


if __name__ == '__main__':
    main()