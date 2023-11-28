import numpy as np
import matplotlib.pyplot as plt

def simulate_uv_plane(sky_model, baselines, uv_size):
    # Initialize the u-v plane
    uv_plane = np.zeros(uv_size, dtype=np.complex128)

    # Sky model dimensions
    sky_height, sky_width = sky_model.shape
    l_max = m_max = 1  # Assuming the field of view is from -1 to 1 in l and m directions

    # Iterate over each baseline
    for dx, dy in baselines:
        # Convert baseline to u-v coordinates (simplified)
        u = dx / wavelength
        v = dy / wavelength

        # Simulate the measurement for this baseline
        for i in range(sky_height):
            for j in range(sky_width):
                l = -l_max + 2 * l_max * i / sky_height
                m = -m_max + 2 * m_max * j / sky_width
                uv_plane[int(u), int(v)] += sky_model[i, j] * np.exp(-2j * np.pi * (u * l + v * m))

    return uv_plane

# Parameters
wavelength = 0.1  # Wavelength of observation in meters
baseline_length = 10  # Maximum baseline length in meters

# Define a simple 10x10 sky model
sky_model = np.zeros((10, 10))
sky_model[5, 5] = 1  # A bright source at the center

# Define baselines (example: a simple grid of telescopes)
baselines = [(dx, dy) for dx in range(baseline_length) for dy in range(baseline_length)]

# Simulate the u-v plane
uv_plane = simulate_uv_plane(sky_model, baselines, uv_size=(baseline_length, baseline_length))

# Inverse Fourier Transform to recreate the image
recreated_image = np.fft.ifft2(np.fft.ifftshift(uv_plane))
recreated_image = np.abs(recreated_image)

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(sky_model, cmap='gray')
axs[0].set_title('Original Sky Model')
axs[1].imshow(np.log(np.abs(uv_plane) + 1e-5), cmap='gray')
axs[1].set_title('u-v Plane')
axs[2].imshow(recreated_image, cmap='gray')
axs[2].set_title('Recreated Image')

plt.show()
