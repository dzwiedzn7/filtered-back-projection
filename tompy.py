from __future__ import division
#import scipy
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq
from scipy.interpolate import interp1d
from gui import imagename

def grey_scale(photo):
    return photo[:,:,0] * 0.299 + photo[:,:,1] * 0.587 + photo[:,:,2] * 0.114

def radon_transform(image, steps):
    radon = np.zeros((steps, len(image)), dtype='float64')
    for s in range(steps):
        rotation = misc.imrotate(image, -s*180/steps).astype('float64')
        radon[:,s] = sum(rotation)
    return radon

def sinogram_circle_to_square(sinogram):
    diagonal = int(np.ceil(np.sqrt(2) * sinogram.shape[0]))
    pad = diagonal - sinogram.shape[0]
    old_center = sinogram.shape[0] // 2
    new_center = diagonal // 2
    pad_before = new_center - old_center
    pad_width = ((pad_before, pad - pad_before), (0, 0))
    return np.pad(sinogram, pad_width, mode='constant', constant_values=0)

def angle(i): return (np.pi*i)/N

def iradon_transform(radon_image, theta=None,interpolation='linear'):
    output_size = radon_image.shape[0]
    radon_image = sinogram_circle_to_square(radon_image)
    th = (np.pi / 180.0) * theta
    # resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = \
        max(64, int(2 ** np.ceil(np.log2(2 * radon_image.shape[0]))))
    pad_width = ((0, projection_size_padded - radon_image.shape[0]), (0, 0))
    img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)
    f = fftfreq(projection_size_padded).reshape(-1, 1)   # digital frequency
    omega = 2 * np.pi * f                                # angular frequency
    fourier_filter = 2 * np.abs(f)                       # ramp filter
    projection = fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(ifft(projection, axis=0))
    radon_filtered = radon_filtered[:radon_image.shape[0], :]
    reconstructed = np.zeros((output_size, output_size))
    mid_index = radon_image.shape[0] // 2
    [X, Y] = np.mgrid[0:output_size, 0:output_size]
    xpr = X - int(output_size) // 2
    ypr = Y - int(output_size) // 2
    # Reconstruct image by interpolation
    interpolation_types = ('linear', 'nearest', 'cubic')
    if interpolation not in interpolation_types:
        raise ValueError("Unknown interpolation: %s" % interpolation)
    for i in range(len(theta)):
        t = ypr * np.cos(th[i]) - xpr * np.sin(th[i])
        x = np.arange(radon_filtered.shape[0]) - mid_index
        if interpolation == 'linear':
            backprojected = np.interp(t, x, radon_filtered[:, i],
                                      left=0, right=0)
        else:
            interpolant = interp1d(x, radon_filtered[:, i], kind=interpolation,
                                   bounds_error=False, fill_value=0)
            backprojected = interpolant(t)
        reconstructed += backprojected
    radius = output_size // 2
    reconstruction_circle = (xpr ** 2 + ypr ** 2) <= radius ** 2
    reconstructed[~reconstruction_circle] = 0.

    return reconstructed * np.pi / (2 * len(th))








image = misc.imread(imagename,flatten=True).astype('float64') # TODO zapytanie o obrazek gui
if len(image.shape) == 3:
    image = grey_scale(image)
image = misc.imresize(image,(220,220))
sinogram = radon_transform(image, 220)

#S=220


theta = np.linspace(0., 180., max(image.shape), endpoint=False)
reconstruction_fbp = iradon_transform(sinogram,theta=theta,interpolation='cubic')


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(image, cmap=plt.cm.Greys_r)
ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

fig.tight_layout()
plt.show()

error = reconstruction_fbp - image
print('FBP rms reconstruction error: %.3g' % np.sqrt(np.mean(error**2)))

imkwargs = dict(vmin=-0.2, vmax=0.2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                               sharex=True, sharey=True,
                               subplot_kw={'adjustable': 'box-forced'})
ax1.set_title("Reconstruction\nFiltered back projection")
ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
ax2.set_title("Reconstruction error\nFiltered back projection")
ax2.imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, **imkwargs)
plt.show()
