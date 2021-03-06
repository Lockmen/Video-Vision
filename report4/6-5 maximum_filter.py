import cv2

def minimumBoxFilter(n, path_to_image):
  img = cv2.imread(path_to_image)

  # Creates the shape of the kernel
  size = (n, n)
  shape = cv2.MORPH_RECT
  kernel = cv2.getStructuringElement(shape, size)

  # Applies the minimum filter with kernel NxN
  imgResult = cv2.erode(img, kernel)

  # Shows the result
  cv2.namedWindow('Result with n ' + str(n), cv2.WINDOW_NORMAL) # Adjust the window length
  cv2.imshow('Result with n ' + str(n), imgResult)


def maximumBoxFilter(n, path_to_image):
  img = cv2.imread(path_to_image)

  # Creates the shape of the kernel
  size = (n,n)
  shape = cv2.MORPH_RECT
  kernel = cv2.getStructuringElement(shape, size)

  # Applies the maximum filter with kernel NxN
  imgResult = cv2.dilate(img, kernel)

  # Shows the result
  cv2.namedWindow('Result with n ' + str(n), cv2.WINDOW_NORMAL) # Adjust the window length
  cv2.imshow('Result with n ' + str(n), imgResult)


if __name__ == "__main__":
  path_to_image = 'lenna.png'

  print("Test the function minimumBoxFilter()")
  minimumBoxFilter(3, path_to_image)
  minimumBoxFilter(5, path_to_image)
  minimumBoxFilter(7, path_to_image)
  minimumBoxFilter(11, path_to_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

  print("Test the function maximumBoxFilter()")
  maximumBoxFilter(3, path_to_image)
  maximumBoxFilter(5, path_to_image)
  maximumBoxFilter(7, path_to_image)
  maximumBoxFilter(11, path_to_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()