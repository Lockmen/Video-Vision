import numpy as np
import cv2
import math

# Loading Grayscale of an Image
img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

# Making Affine Matrix
# 30 degrees in radians
angle = 10 * (math.pi / 180)
moveX = 100
moveY = -130
rotation_matrix = [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
translation_matrix = [[1, 0, moveX], [0, 1, moveY], [0, 0, 1]]
affine_matrix = np.dot(rotation_matrix, translation_matrix)
np.save('outputP6A.npy', affine_matrix)


def affine_warp(input_image, affMatrix):
    """
    Inputs: Image and arbitrary affine transformation matrix A as introduced in Problem 2.
    Your function must compute and return an output image that results from
    the warping procedure.
    """
    height = len(input_image)
    width = len(input_image[0])

    # Output Image
    res = np.zeros((height, width, 1), dtype='uint8')

    # get color img[y][x] --> transfer to new location in res[y'][x']

    for r in range(height):
        for c in range(width):
            # calculate the affine transformation of that point
            currentCoordinate = np.array([[c], [r], [1]])
            transformedCoordinates = np.dot(affMatrix, currentCoordinate)
            newX = int(transformedCoordinates[0, 0])
            newY = int(transformedCoordinates[1, 0])
            if newX < height and newY < width and newX >= 0 and newY >= 0:
                res[newY, newX] = input_image[r, c]

    return res


transformed_image = affine_warp(img, affine_matrix)
cv2.namedWindow('INPUT')
cv2.imshow('INPUT', img)
cv2.imshow('AFFINE_TRANSFORMATION_OUTPUT', transformed_image)
cv2.imwrite('outputP6.png', transformed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()