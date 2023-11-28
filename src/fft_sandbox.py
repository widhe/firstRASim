import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift

# Parameters
width, height = 100, 100  # Size of the 2D data
frequency_x = 5           # Frequency in x-direction
frequency_y = 5           # Frequency in y-direction

# Generate grid of x and y values
x = np.linspace(0, 2 * np.pi, width)
y = np.linspace(0, 2 * np.pi, height)
xx, yy = np.meshgrid(x, y)

# Create 2D wave
data = np.sin(frequency_x * xx)
#data = np.sin(frequency_x * xx) + np.sin(frequency_y * yy)

# Plot
plt.imshow(data, cmap='gray', extent=[0, width, 0, height])
plt.colorbar()
plt.title('2D Single Frequency Wave')
plt.show()

fft_data = fftshift(fft2(data))

plt.imshow(np.abs(fft_data), cmap='gray', extent=[0, width, 0, height])
plt.show()
