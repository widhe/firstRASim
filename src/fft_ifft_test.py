from numpy.fft import fft2, ifft2, fftshift
from ra_functions_chat_gpt import *

from PIL import Image


image_size = (101, 101)  # Size of the output image (x,y)
cmap = 'plasma'

dict_img = load_image_and_resize('empty.jpg', image_size)

input_image_full = dict_img['image_full']
input_image = dict_img['image']

input_image[:] = 0
x_f = 0
y_f = 0
input_image[50 + y_f, 50 + x_f] = 1
input_image[50 - y_f, 50 - x_f] = 1

plt.imshow(input_image, cmap=cmap)
plt.colorbar(label='Intensisty')
plt.title('Image')
plt.show()

visibility_like = fftshift(fft2(input_image))

plt.imshow(np.abs(visibility_like), cmap=cmap)
plt.colorbar(label='Magnitude')
plt.title('Magnitude visibility')
plt.show()

plt.imshow(np.unwrap(np.angle(visibility_like)), cmap=cmap)
plt.title('Image')
plt.colorbar(label='Phase')
plt.title('Phase visibility')
plt.show()


# Then, inverse Fourier Transform this to get the output image
output_image = ifft2(visibility_like)

# Take the absolute value to get intensity
output_image = np.abs(output_image)

plt.imshow(output_image, cmap=cmap)
plt.colorbar(label='Intensisty')
plt.title('Reconstructed Image')
plt.show()