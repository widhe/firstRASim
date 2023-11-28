import numpy as np
import matplotlib.pyplot as plt

# Define a simple 10x10 sky model
# For simplicity, let's create a sky model with a single source in the center
sky_model = np.zeros((100, 100))
sky_model[51, 50] = 1  # A bright source at the center
cmap = 'plasma'

# Fourier Transform of the sky model (to u-v plane)
uv_plane = np.fft.fftshift(np.fft.fft2(sky_model))

# Inverse Fourier Transform to recreate the image
recreated_image = np.fft.ifft2(np.fft.ifftshift(uv_plane))
recreated_image = np.abs(recreated_image)  # Taking the absolute value

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
im0 = axs[0].imshow(sky_model, cmap=cmap)
plt.colorbar(im0, ax=axs[0])
axs[0].set_title('Original Sky Model')
im1 = axs[1].imshow(np.abs(uv_plane), cmap=cmap)
plt.colorbar(im1, ax=axs[1])
axs[1].set_title('u-v Plane')
im2 = axs[2].imshow(recreated_image, cmap=cmap)
plt.colorbar(im2, ax=axs[2])
axs[2].set_title('Recreated Image')

plt.show()
