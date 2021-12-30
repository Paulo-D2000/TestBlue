import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn

#Functions
fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift

# Mesh on the square [0,1)x[0,1)
x = np.linspace( -4, 4, 256)     # columns (Width)
y = np.linspace( -4, 4, 256)     # rows (Height)

#Image Grid
[X,Y] = np.meshgrid(x,y)

#Distance from center
r = np.sqrt(X**2+Y**2)

#Bessel Function
z = spherical_jn(0,r)

#"Filter" ? dist. from center
filt = r

#Bessel * white noise dft * filt
b_dft = (z * np.fft.fftshift(np.fft.fft2(10.0+np.random.random(z.shape)))) * filt

#IFT
bnoise = np.fft.ifftshift(np.fft.ifft2(b_dft))

#Plot the Frequency domain "Blue" noise
plt.imshow(20*np.log10(np.abs(b_dft)))

#Plot the "Filter"
plt.figure()
plt.imshow(np.abs(filt))

#Plot the output
plt.figure()
plt.imshow(np.abs(bnoise))

plt.show()
