import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from matplotlib.colors import LogNorm
import cv2
import time

start = time.time()
#Load input image
image_source = cv2.imread('C:/FaksGit/FourierFilter/TestImages/man.png')
gray_image = cv2.cvtColor(image_source, cv2.COLOR_BGR2GRAY)

#Plot input image
plt.figure()
plt.imshow(gray_image, plt.cm.gray)
plt.xticks([]), plt.yticks([])
plt.title("Original image")

#Return the two-dimensional discrete Fourier transform of the 2-D argument x.
image_fft = fftpack.fft2(gray_image)

#Logaritmic map
def show_spectrum(image_fft):
    plt.imshow(np.abs(image_fft), norm=LogNorm(vmin=5))
    plt.colorbar()  #Add colorbar

#Plot FT input image
plt.figure()
show_spectrum(image_fft)
plt.title("Fourier transform")

keep_fraction = 0.3  #keep fraction (u oba smijera)
image_fft2 = image_fft.copy()
row, col = image_fft2.shape  #get the current shape of an array

#Set on zero all rows with index between row*keep_fraction and row*(1-keep_fraction)
image_fft2[int(row*keep_fraction):int(row*(1-keep_fraction))] = 0
#Similar for columns
image_fft2[:, int(col*keep_fraction):int(col*(1-keep_fraction))] = 0

#Plot spectrum
plt.figure()
show_spectrum(image_fft2)
plt.title('Filtered Spectrum')

#Return inverse two-dimensional discrete Fourier transform of arbitrary type sequence x
image_new = fftpack.ifft2(image_fft2).real

end = time.time()
print("Time:" ,end - start)

fig = plt.figure(figsize=(20,10))
plt.imshow(image_new, plt.cm.gray)
plt.xticks([]), plt.yticks([])
plt.title('Reconstructed Image FFT')
fig.savefig('baboon_gauss60.3FFT.png')


