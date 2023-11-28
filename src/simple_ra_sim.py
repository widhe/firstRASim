from ra_functions_chat_gpt import *
import matplotlib.pyplot as plt
import numpy as np

def simple_ra_sim(image_path, num_antennas, antenna_placement_method, antenna_area_side, image_size = (101, 101),
                  test_val=None, test_baselines=None, sw_plot=False):

    cmap = 'plasma'
    sw_fill_uv_plane = False
    area_side = antenna_area_side  # the side of the area to place antennas [meters]

    x_plts = 4
    y_plts = 3
    sb_plt = 1


    dict_img = load_image_and_resize(image_path, image_size)

    input_image_full = dict_img['image_full']
    input_image = dict_img['image']

    image_center = (np.rint((image_size[0] - 1) / 2).astype(int), np.rint((image_size[1] - 1) / 2).astype(int))

    if test_val is not None:
        input_image[:] = 0
        x_f = test_val[0]
        y_f = test_val[1]
        input_image[image_center[0] + y_f, image_center[1] + x_f] = 1
        input_image[image_center[0] - y_f, image_center[1] - x_f] = 1

    visibility_like = np.fft.fft2(input_image)

    if sw_fill_uv_plane:
        antenna_positions = []
    else:
        antenna_positions = place_antennas(num_antennas, area_side, antenna_placement_method)

    if sw_fill_uv_plane:
        # Full uv-coverage
        x_half = np.rint(image_size[0]/2).astype(int)
        y_half = np.rint(image_size[0]/2).astype(int)
        baselines = list(itertools.product(range(-x_half, x_half+1), range(-y_half, y_half+1)))
    else:
        if test_baselines is None:
            # uv-coverage from antennas
            baselines = generate_baselines(antenna_positions)
            baselines = grid_baselines(baselines)
        else:
            baselines = test_baselines


    visibilities = populate_visiblility_array(visibility_like, baselines)

    output_image_perfect = ifft2(visibility_like)
    output_image = ifft2(visibilities)

    # Take the absolute value to get intensity
    output_image_perfect = np.abs(output_image_perfect)
    output_image = np.abs(output_image)

    if sw_plot:
        plt.subplot(y_plts, x_plts, sb_plt)
        sb_plt += 1
        plt.imshow(input_image_full, cmap=cmap)
        plt.title('Input image full size')

        plt.subplot(y_plts, x_plts, sb_plt)
        sb_plt += 1
        plt.imshow(input_image, cmap=cmap)
        plt.title('Input image')

        plt.subplot(y_plts, x_plts, sb_plt)
        sb_plt += 1
        plt.imshow(np.abs(visibility_like))
        plt.colorbar()
        plt.title('Visibilities (abs)')

        plt.subplot(y_plts, x_plts, sb_plt)
        sb_plt += 1
        #plt.imshow(np.angle(visibility_like))
        plt.imshow(np.unwrap(np.angle(visibility_like)))
        plt.colorbar()
        plt.title('Visibilities (phase)')

        plt.subplot(y_plts, x_plts, sb_plt)
        sb_plt += 1
        plot_antenna_positions(antenna_positions)

        plt.subplot(y_plts, x_plts, sb_plt)
        sb_plt += 1
        plot_uv_coverage(baselines)

        plt.subplot(y_plts, x_plts, sb_plt)
        sb_plt += 1
        plt.imshow(np.abs(visibilities))
        plt.colorbar()
        plt.title('Visibilities (abs)')

        plt.subplot(y_plts, x_plts, sb_plt)
        sb_plt += 1
        #plt.imshow(np.angle(visibilities))
        plt.imshow(np.unwrap(np.angle(visibilities)))
        plt.colorbar()
        plt.title('Visibilities (phase)')


        plt.subplot(y_plts, x_plts, sb_plt)
        sb_plt += 1
        plt.imshow(output_image_perfect, cmap=cmap)
        plt.title('Output image (perfect)')

        plt.subplot(y_plts, x_plts, sb_plt)
        sb_plt += 1
        plt.imshow(output_image, cmap=cmap)
        plt.title('Output image')

        plt.show()

    return {'input_image': input_image,
            'antenna_positions': antenna_positions,
            'baselines': baselines,
            'output_image': output_image}
