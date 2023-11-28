import matplotlib.pyplot as plt
from simple_ra_sim import simple_ra_sim
import numpy as np

np.random.seed(12)

plt.rcParams['axes.titlesize'] = 8  # For subplot titles
plt.rcParams['axes.labelsize'] = 6 # For x and y labels
plt.rcParams['xtick.labelsize'] = 6  # For x-axis tick labels
plt.rcParams['ytick.labelsize'] = 6  # For y-axis tick labels
plt.rcParams['lines.markersize'] = 2

antenna_placement_method = 'circle'
#antenna_placement_method = 'random_uniform'
num_antennas = 20
antenna_area_side = 100
sw_test_baselines = False

if sw_test_baselines:
    test_baselines = []
    for u in range(-5,6):
        for v in range (0, 6):
            test_baselines.append(np.array([u,v]))
else:
    test_baselines = None


fig = plt.figure(figsize=(10, 10))
#simple_ra_sim('big_dipper.jpg', num_antennas, antenna_placement_method, antenna_area_side, image_size=(11,11),
#              test_val=(1, 0), test_baselines=test_baselines, sw_plot=True)
simple_ra_sim('big_dipper.jpg', num_antennas, antenna_placement_method, antenna_area_side, image_size=(101,101),
              test_val=None, test_baselines=test_baselines, sw_plot=True)

fig.suptitle(f'Num antennas = {num_antennas}, Antenna placement method = {antenna_placement_method}, area_size = {antenna_area_side} m')
plt.subplots_adjust(wspace=0.4)
