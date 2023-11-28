from ra_functions_chat_gpt import *
import matplotlib.pyplot as plt
import numpy as np


image_path = 'big_dipper.jpg'
num_antennas = 100
antenna_area_side = 100  # meters
antenna_placement_method = 'random_uniform'
#antenna_placement_method = 'circle'
image_size = (11, 11)  # Size of the output image
fov = 1.0 # Field of view in degrees
image_center_ra_dec = (180, 45) # Center RA and Dec in degrees
wavelength = 0.21 #e-9  # Wavelength of observation in meters
#wavelength = 0.705  #e-9  # Wavelength of observation in meters
#wavelength = 0.705/2  #e-9  # Wavelength of observation in meters
wavelength = 0.0125  #e-9  # Wavelength of observation in meters
wavelength = np.cos(fov/360*2*np.pi)*2*np.pi/(image_size[0])/pow(2,0)  #e-9  # Wavelength of observation in meters
#wavelength = 0.705/4  #e-9  # Wavelength of observation in meters
test_val = (1, 0)
sw_test_baselines = True
#test_val = None
cmap = 'plasma'

if sw_test_baselines:
    test_baselines = []
    for u in range(-5,6):
        #for v in range (-10, 11):
        for v in range(0, 6):
            test_baselines.append(np.array([u,v]))
else:
    test_baselines = None


np.random.seed(12)

plt.rcParams['axes.titlesize'] = 8  # For subplot titles
plt.rcParams['axes.labelsize'] = 6 # For x and y labels
plt.rcParams['xtick.labelsize'] = 6  # For x-axis tick labels
plt.rcParams['ytick.labelsize'] = 6  # For y-axis tick labels
plt.rcParams['lines.markersize'] = 2

x_plts = 4
y_plts = 4
sb_plt = 1

fig = plt.figure(figsize=(10, 10))

dict_img = load_image_and_resize(image_path, image_size)

input_image_full = dict_img['image_full']
input_image = dict_img['image']

image_center = (np.rint((image_size[0]-1)/2).astype(int), np.rint((image_size[1]-1)/2).astype(int))

if test_val is not None:
    input_image[:] = 0
    x_f = test_val[0]
    y_f = test_val[1]
    input_image[image_center[0] + y_f, image_center[1] + x_f] = 1
    input_image[image_center[0] - y_f, image_center[1] - x_f] = 1

plt.subplot(y_plts, x_plts, sb_plt)
sb_plt += 1
plt.imshow(input_image, cmap=cmap)


sky_model = convert_image_to_sky_model(input_image, fov, image_center_ra_dec)
plt.subplot(y_plts, x_plts, sb_plt)
sb_plt += 1
plot_sky_model(sky_model)

# Create an antenna layout
antenna_positions = place_antennas(num_antennas, antenna_area_side, antenna_placement_method)

plt.subplot(y_plts, x_plts, sb_plt)
sb_plt += 1
plot_antenna_positions(antenna_positions)

if test_baselines is None:
    baselines = generate_baselines(antenna_positions)
else:
    baselines = test_baselines

plt.subplot(y_plts, x_plts, sb_plt)
sb_plt += 1
plot_uv_coverage(baselines)

visibilities = simulate_visibilities(sky_model, baselines, wavelength)
plt.subplot(y_plts, x_plts, sb_plt)
sb_plt += 1
plot_visibilities_uv(baselines, visibilities, 'abs')
plt.subplot(y_plts, x_plts, sb_plt)
sb_plt += 1
plot_visibilities_uv(baselines, visibilities, 'phase')

visibility_grid, sky_brightness_image = create_sky_brightness_image(visibilities, baselines, image_size)

# Display the result
plt.subplot(y_plts, x_plts, sb_plt)
sb_plt += 1
plt.imshow(np.abs(visibility_grid))
plt.colorbar()
plt.title('Abs Visibility gridded')

plt.subplot(y_plts, x_plts, sb_plt)
sb_plt += 1
plt.imshow(np.angle(visibility_grid))
plt.colorbar()
plt.title('Phase Visibility gridded')

plt.subplot(y_plts, x_plts, sb_plt)
sb_plt += 1
plt.imshow(sky_brightness_image, cmap=cmap)
plt.title('Sky brightenss map')

#plt.figure()
#plt.imshow(sky_brightness_image, cmap='gray')
plt.subplots_adjust(hspace=0.4)

plt.show()
