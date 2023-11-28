import matplotlib.pyplot as plt
from simple_ra_sim import simple_ra_sim
from ra_functions_chat_gpt import *
import os

match 4:
    case 1:
        antenna_placement_method = "circle"
    case 2:
        antenna_placement_method = "random_uniform"
    case 3:
        antenna_placement_method = "spiral"
    case 4:
        antenna_placement_method = "spiral_arms"

antenna_area_side = 100

match 8:
    case 1:
        image_path = 'simple circle.jpg'
    case 2:
        image_path = 'simple circle black background.jpg'
    case 3:
        image_path = 'half_half.jpg'
    case 4:
        image_path = 'vertical_stripes.jpg'
    case 5:
        image_path = 'horizontal_stripes.jpg'
    case 6:
        image_path = 'four_pointy_star.jpg'
    case 7:
        image_path = 'complex_star.jpg'
    case 8:
        image_path = 'big_dipper.jpg'

num_antennas_list = [10, 50, 100, 300, 500]
cmap = 'plasma'

plt.rcParams['axes.titlesize'] = 8  # For subplot titles
plt.rcParams['axes.labelsize'] = 6 # For x and y labels
plt.rcParams['xtick.labelsize'] = 6  # For x-axis tick labels
plt.rcParams['ytick.labelsize'] = 6  # For y-axis tick labels
plt.rcParams['lines.markersize'] = 2

x_plts = 4
y_plts = len(num_antennas_list)
sb_plt = 1

fig = plt.figure(figsize=(8, 10))

for num_antennas in num_antennas_list:
    res_dict = simple_ra_sim(image_path, num_antennas, antenna_placement_method, antenna_area_side)

    plt.subplot(y_plts, x_plts, sb_plt)
    sb_plt += 1
    plt.imshow(res_dict['input_image'], cmap=cmap)
    plt.title('Input image')

    plt.subplot(y_plts, x_plts, sb_plt)
    sb_plt += 1
    plot_antenna_positions(res_dict['antenna_positions'])

    plt.subplot(y_plts, x_plts, sb_plt)
    sb_plt += 1
    plot_uv_coverage(res_dict['baselines'])

    plt.subplot(y_plts, x_plts, sb_plt)
    sb_plt += 1
    plt.imshow(res_dict['output_image'], cmap=cmap)
    plt.title('Output image')

fig.suptitle(f'Antenna placement method = {antenna_placement_method}, area_size = {antenna_area_side} m')
plt.subplots_adjust(wspace=0.4)
fig.savefig(f'../figures/{os.path.basename(image_path)}-{antenna_placement_method}-{antenna_area_side}.png', dpi=300, bbox_inches='tight')
plt.show()


