import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
from numpy.fft import ifft2, fftshift
import itertools
from tqdm import tqdm


def plot_sky_model(sky_model):
    ras = [ra for ra, dec, flux in sky_model]
    decs = [dec for ra, dec, flux in sky_model]
    fluxes = [flux for ra, dec, flux in sky_model]

    plt.scatter(ras, decs, s=np.sqrt(fluxes), c=fluxes, cmap='plasma')  # Adjust size and color based on flux
    plt.xlabel('Right Ascension')
    plt.ylabel('Declination')
    plt.title('Sky Model Visualization')
    #plt.colorbar(label='Flux')


def plot_visibilities(visibilities):
    real_parts = [np.real(v) for v in visibilities]
    imag_parts = [np.imag(v) for v in visibilities]

    plt.scatter(real_parts, imag_parts, marker='.')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Visibility Plot')
    plt.grid(True)


def plot_visibilities_uv(baselines, visibilities, type):
    u_coords = [u for u, _ in baselines]
    v_coords = [v for _, v in baselines]
    magnitudes = [np.abs(v) for v in visibilities]
    phases = [np.angle(v) for v in visibilities]

    if type == 'abs':
        # Plot Magnitude
        plt.scatter(u_coords, v_coords, c=magnitudes, cmap='plasma')
        #plt.colorbar(label='Magnitude')
        plt.xlabel('u [m]')
        plt.ylabel('v [m]')
        plt.title('Visibility Magnitude')
    else:
        # Plot Phase
        plt.scatter(u_coords, v_coords, c=phases, cmap='plasma')
        #plt.colorbar(label='Phase (Radians)')
        plt.xlabel('u [m]')
        plt.ylabel('v [m]')
        plt.title('Visibility Phase')

def plot_uv_coverage(baselines):
    u_coords = [u for u, v in baselines]
    v_coords = [v for u, v in baselines]

    #plt.figure(figsize=(8, 8))
    plt.scatter(u_coords, v_coords, marker='.')
    plt.scatter(-np.array(u_coords), -np.array(v_coords), marker='.')  # Plotting the conjugate points
    plt.title(f'UV Coverage ({2*len(baselines)})')
    plt.grid(True)
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.gca().set_aspect('equal', 'box')


def plot_antenna_positions(antenna_positions):
    x_coords = [x for x, y in antenna_positions]
    y_coords = [y for x, y in antenna_positions]

    #plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, marker='x')
    plt.title(f'Antenna postions ({len(antenna_positions)})')
    plt.grid(True)
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    plt.gca().set_aspect('equal', 'box')



def load_image(image_path):
    # Load the image file
    img = Image.open(image_path)

    # Convert to grayscale if it's a color image
    if img.mode != 'L':
        img = img.convert('L')

    # Convert the image to a NumPy array
    image_data = np.array(img)

    # Normalize the image data to range between 0 and 1
    normalized_image_data = image_data / 255.0

    return normalized_image_data

def convert_image_to_sky_model(image_data, fov, image_center_ra_dec):
    num_rows, num_cols = image_data.shape
    sky_model = []

    for i in range(num_rows):
        for j in range(num_cols):
            # Calculate RA and Dec for each pixel
            ra = image_center_ra_dec[0] + (j - num_cols / 2) * fov / num_cols
            dec = image_center_ra_dec[1] + (i - num_rows / 2) * fov / num_rows

            # Assign flux density based on pixel intensity
            flux_density = image_data[i, j]

            sky_model.append((ra, dec, flux_density))

    return sky_model


def generate_baselines(antenna_positions):
    # Create all pairs of antennas
    baseline_pairs = itertools.combinations(antenna_positions, 2)

    # Calculate the baselines as differences between antenna positions
    baselines = [np.subtract(p2, p1) for p1, p2 in baseline_pairs]

    return baselines


def simulate_visibilities(sky_model, baselines, wavelength):
    visibilities = np.zeros(len(baselines), dtype=complex)

    #for i, (u, v) in enumerate(tqdm(baselines)):
    for i, (u, v) in enumerate(baselines):
        for ra, dec, flux in sky_model:
            # Convert RA, Dec to radians for calculations
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)

            # Calculate baseline components in wavelengths
            u_wavelength = u / wavelength
            v_wavelength = v / wavelength

            # Calculate phase shift
            phase_shift = 2 * np.pi * \
                          (u_wavelength * np.cos(dec_rad) * np.cos(ra_rad)
                           + v_wavelength * np.cos(dec_rad) * np.sin(ra_rad))

            # Add contribution of this source to the visibility
            visibilities[i] += flux * np.exp(-1j * phase_shift)

    return visibilities


def create_sky_brightness_image(visibilities, baselines, grid_size):
    # Initialize a grid of zeros
    visibility_grid = np.zeros(grid_size, dtype=complex)

    # Calculate the grid center
    grid_center = (grid_size[0] // 2, grid_size[1] // 2)

    for i, (u, v) in enumerate(baselines):
        # Map (u, v) to grid indices
        u_idx = grid_size[0] - 1 - int(grid_center[0] + u)
        v_idx = int(grid_center[1] + v)

        if 0 <= u_idx < grid_size[0] and 0 <= v_idx < grid_size[1]:
            #visibility_grid[v_idx, u_idx] += visibilities[i]
            visibility_grid[v_idx, u_idx] = visibilities[i] # Test

        # Also add the conjugate visibility
        u_idx_conj = grid_size[0] - 1 - int(grid_center[0] - u)
        v_idx_conj = int(grid_center[1] - v)

        if 0 <= u_idx_conj < grid_size[0] and 0 <= v_idx_conj < grid_size[1]:
            #visibility_grid[v_idx_conj, u_idx_conj] += np.conjugate(visibilities[i])
            visibility_grid[v_idx_conj, u_idx_conj] = np.conjugate(visibilities[i]) # Test

        pass

    # Perform an inverse Fourier transform
    visibility_grid = fftshift(visibility_grid)
    sky_brightness_image = ifft2(visibility_grid)

    # Use fftshift to center the zero frequencies
    sky_brightness_image = fftshift(sky_brightness_image)

    # Take the absolute value to get intensity
    sky_brightness_image = np.abs(sky_brightness_image)

    return visibility_grid, sky_brightness_image


def generate_spiral_antenna_positions(steps, step_size=1.0, angle_step=0.1):
    """
    Generate a list of antenna positions in a spiral pattern.
    :param steps: Number of points (antennas) to generate.
    :param step_size: The rate at which the spiral expands.
    :param angle_step: The angle step for each iteration in radians.
    :return: A list of (x, y) coordinates.
    """
    positions = []
    theta = 0.0

    for _ in range(steps):
        r = step_size * theta
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        positions.append((x, y))
        theta += angle_step

    return positions

import math

def generate_six_arm_spiral_antenna_positions(steps_per_arm, arms=6, step_size=1.0, angle_step=0.1):
    """
    Generate antenna positions in a six-arm spiral pattern.
    :param steps_per_arm: Number of points (antennas) to generate per arm.
    :param arms: Number of spiral arms.
    :param step_size: The rate at which the spiral expands.
    :param angle_step: The angle step for each iteration in radians.
    :return: A list of (x, y) coordinates.
    """
    positions = []
    angle_offset = 2 * math.pi / arms

    for arm in range(arms):
        start_angle = arm * angle_offset
        theta = start_angle

        for _ in range(steps_per_arm):
            r = step_size * (theta - start_angle)  # Reset radius for each arm
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            positions.append((x, y))
            theta += angle_step

    return positions


def place_antennas(num_antennas, area_side, method):
    match method:
        case 'circle':
            antenna_positions = [(area_side/2 * np.cos(2 * np.pi * i / num_antennas),
                                  area_side/2 * np.sin(2 * np.pi * i / num_antennas))
                                 for i in range(num_antennas)]
        case 'random_uniform':
            antenna_positions = [(np.random.uniform(-area_side/2, area_side/2),
                                  np.random.uniform(-area_side/2, area_side/2))
                                 for _ in range(num_antennas)]

        case 'spiral':
            angle_step = 0.1
            antenna_positions = generate_spiral_antenna_positions(num_antennas,
                                                                  step_size=area_side/2/(num_antennas*angle_step),
                                                                  angle_step=angle_step)

        case 'spiral_arms':
            angle_step = 0.1
            num_arms = 6
            antenna_positions = generate_six_arm_spiral_antenna_positions(round(num_antennas/num_arms),
                                                                          num_arms,
                                                                          step_size=area_side/2/(num_antennas/num_arms*angle_step),
                                                                          angle_step=angle_step)
        case _:
            raise Exception(f"Unknown method {method}")

    return antenna_positions

def load_image_and_resize(image_path, image_size):
    # Load the image file
    img = Image.open(image_path)

    # Convert to grayscale if it's a color image
    if img.mode != 'L':
        img = img.convert('L')

    # Convert the image to a NumPy array
    image_data_full = np.array(img)
    image_data = np.array(img.resize(image_size))
    # Normalize the image data to range between 0 and 1
    normalized_image_data_full = image_data_full / 255.0
    normalized_image_data = image_data / 255.0

    return {'image_full': normalized_image_data_full, 'image': normalized_image_data}


def grid_baselines(baselines):
    transformed_list = []
    for coords in tqdm(baselines):
        # Shift the range from [-50, 50] to [0, 100]
        #shifted_coords = coords + 50
        shifted_coords  = coords

        # Round to nearest integer and convert to int
        int_coords = np.rint(shifted_coords).astype(int)

        transformed_list.append(int_coords)

    return transformed_list


def populate_visiblility_array(full_array, coordinates):
    sparse_size = full_array.shape
    # Initialize a sparse array with zeros
    sparse_array = np.zeros(sparse_size, dtype=full_array.dtype)

    # Populate the sparse array with the values from the full array
    for x, y in tqdm(coordinates):
        # Ensure the coordinates are within the bounds of the full array
        x_half = np.rint((sparse_size[0]-1) / 2).astype(int)
        y_half = np.rint((sparse_size[1]-1) / 2).astype(int)
        xc = (-x + x_half).astype(int)
        yc = (sparse_size[1] - 1 - (-y + y_half)).astype(int)
        x = (x + x_half).astype(int)
        y = (sparse_size[1] - 1 - (y + y_half)).astype(int)
        if 0 <= x < full_array.shape[0] and 0 <= y < full_array.shape[1]:
            sparse_array[y, x] = full_array[y, x]
        if 0 <= xc < full_array.shape[0] and 0 <= yc < full_array.shape[1]:
            sparse_array[yc, xc] = full_array[yc, xc]
        pass

    return sparse_array
