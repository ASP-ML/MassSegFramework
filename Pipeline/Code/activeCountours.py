import logging
import numpy as np
from imageio import imread
from matplotlib import pyplot as plt
from PIL import Image

import morphsnakes as ms

def visual_callback_2d(background, fig=None):
    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(background, cmap=plt.cm.gray)

    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
    plt.pause(0.001)

    def callback(levelset):
        if ax1.collections:
            del ax1.collections[0]
        ax1.contour(levelset, [0.5], colors='r')
        ax_u.set_data(levelset)
        fig.canvas.draw()
        plt.pause(0.001)

    return callback

'''
def visual_callback_3d(fig=None, plot_each=1):
    from mpl_toolkits.mplot3d import Axes3D
    # PyMCubes package is required for `visual_callback_3d`
    try:
        import mcubes
    except ImportError:
        raise ImportError("PyMCubes is required for 3D `visual_callback_3d`")

    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')
    plt.pause(0.001)
    counter = [-1]

    def callback(levelset):

        counter[0] += 1
        if (counter[0] % plot_each) != 0:
            return

        if ax.collections:
            del ax.collections[0]

        coords, triangles = mcubes.marching_cubes(levelset, 0.5)
        ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                        triangles=triangles)
        plt.pause(0.1)

    return callback
'''

def rgb2gray(img):
    #Convert a RGB image to gray scale
    return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

def CV_active_contour(x, y, input_path, output_pathCV):
    logging.info('Running: active_contour (MorphACWE)...')

    # Load the image.
    img = imread(input_path)[..., 0] / 255.0

    # Initialization of the level-set.
    init_ls = ms.circle_level_set(img.shape, (y, x), 40)

    # Morphological Chan-Vese (or ACWE)
    array = ms.morphological_chan_vese(img, iterations=436,
                               init_level_set=init_ls,
                               smoothing=3, lambda1=1, lambda2=1)

    #Save mask
    image_u = Image.fromarray((array * 255).astype(np.uint8), mode='L')
    image_u.save(output_pathCV)
    
def pipLine(centerPoint, pathFiltered, outputFolder):
    logging.basicConfig(level=logging.DEBUG)

    input_path = pathFiltered
    filename = pathFiltered.split('d/')[1].split('_f')[0]
    output_pathCV = outputFolder + 'mask_CVBWCF_' + filename + '.jpg'
    CV_active_contour(centerPoint[0], centerPoint[1], input_path, output_pathCV)

    logging.info("Done.")
