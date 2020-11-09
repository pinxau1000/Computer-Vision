# ------------------------------------------------------------------------------
#                                   IMPORTS
# ------------------------------------------------------------------------------
from sys import version_info
from sys import path as syspath
from os import path
import json

_CURRENT_DIRECTORY = syspath[0]

try:
    import util
    # if you have problems visit:
    # https://gist.github.com/pinxau1000/8817d4ef0ed766c78bac8e6feafc8b47
    # https://github.com/pinxau1000/
except ModuleNotFoundError:
    from urllib import request
    print("'util.py' not found on the same folder as this script!")
    _url_utilpy = "https://gist.githubusercontent.com/pinxau1000/8817d4ef0ed766c78bac8e6feafc8b47/raw/util.py"
    print("Downloading util.py from:\n" + _url_utilpy)
    # https://stackoverflow.com/questions/7243750/download-file-from-web-in-python-3
    request.urlretrieve(_url_utilpy, "util.py")
    print("Downloading finished!")
    import util

try:
    import cv2 as cv
except ModuleNotFoundError:
    util.install("opencv-python")
    import cv2 as cv

try:
    from matplotlib import pyplot as plt
except ModuleNotFoundError:
    util.install("matplotlib")
    from matplotlib import pyplot as plt

try:
    import numpy as np
except ModuleNotFoundError:
    util.install("numpy")
    import numpy as np

try:
    from packaging import version
except ModuleNotFoundError:
    util.install("packaging")
    from packaging import version

try:
    import click
except ModuleNotFoundError:
    util.install("click")
    import click

# ------------------------------------------------------------------------------
#                               REQUIREMENTS CHECK
# ------------------------------------------------------------------------------
assert version_info.major >= 3 and \
       version_info.minor >= 5, \
    "This script requires Python 3.5.0 or above!"

assert version.parse(cv.__version__).major >= 4 and \
       version.parse(cv.__version__).minor >= 4, \
    "This script requires OpenCV 4.4.0 or above!"

assert version.parse(plt.matplotlib.__version__).major >= 3 and \
       version.parse(plt.matplotlib.__version__).minor >= 3, \
    "This script requires MatPlotLib 3.3.0 or above!"

assert version.parse(np.__version__).major >= 1 and \
       version.parse(np.__version__).minor >= 19, \
    "This script requires Numpy 1.19.0 or above!"

assert version.parse(click.__version__).major >= 7 and \
       version.parse(click.__version__).minor >= 1, \
    "This script requires Click 7.1.0 or above!"

# ------------------------------------------------------------------------------
#                           Load Default Pictures
# ------------------------------------------------------------------------------
_PATH_2_DATA = path.join(_CURRENT_DIRECTORY, "../../data/")
_IMG_ORIG_NAME = "img05.jpg"
_IMG_NOISE_NAME = "img05_noise.jpg"
_FULL_PATH_ORIG = path.join(_PATH_2_DATA, _IMG_ORIG_NAME)
_FULL_PATH_NOISE = path.join(_PATH_2_DATA, _IMG_NOISE_NAME)


# ------------------------------------------------------------------------------
#                               Gaussian Filter Sigma
# ------------------------------------------------------------------------------
# PASSED
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_NOISE,
              type=str,
              help="The path to the image")
@click.option("--sigma_x",
              default=str(list(np.arange(0.5, 1.1, 0.5))),
              help="The sigmaX values to be evaluated")
@click.option("--sigma_y",
              default=str(list(np.arange(0.5, 1.1, 0.5))),
              help="The sigmaY values to be evaluated")
@click.option("--crop_corner",
              default=str([480, 110]),
              help="The upper left corner of the crop area as list. The point "
                   "with coordinates x=480 and y=110 is passed as [480,110].")
@click.option("--crop_size",
              default=64,
              type=int,
              help="The size of the crop area")
@click.option("--save",
              default="output_GaussianFilter_Sigma",
              type=str,
              help="The save name(s) of the output figure(s)")
@click.option("--dpi",
              default=None,
              type=int,
              help="Quality of the figure window generated. If None its the "
                   "default 100 dpi.")
@click.option("--num",
              default=None,
              type=int,
              help="Number of the figure window generated. If None its "
                   "cumulative.")
def gaussian_filter_sigma(image, sigma_x, sigma_y, crop_corner, crop_size,
                          save, dpi, num):
    image = util.load_image_RGB(image)
    sigma_x = json.loads(sigma_x)
    sigma_y = json.loads(sigma_y)
    crop_corner = json.loads(crop_corner)

    # Initialize the gaussian_images_sigma as list.
    # Values are assigned on the for-loop.
    gaussian_images_sigmaX = []
    titles_images_X = []
    for sigX in sigma_x:
        gaussian_images_sigmaX.append(cv.GaussianBlur(image, (9, 9),
                                                      sigmaX=sigX,
                                                      sigmaY=0.1))
        titles_images_X.append(f"SigmaX = {sigX}")

    gaussian_images_sigmaY = []
    titles_images_Y = []
    for sigY in sigma_y:
        # SigmaX and SigmaY is 0 so they are calculated from kernel
        gaussian_images_sigmaY.append(cv.GaussianBlur(image, (9, 9),
                                                      sigmaX=0.1,
                                                      sigmaY=sigY))
        titles_images_Y.append(f"SigmaY = {sigY}")

    # Crop filtered images to better see the effect of Sigma.
    gaussian_images_sigmaX_crop = []
    for i in range(len(gaussian_images_sigmaX)):
        gaussian_images_sigmaX_crop.append(gaussian_images_sigmaX[i]
                                           [crop_corner[1]:
                                            crop_corner[1] + crop_size,
                                           crop_corner[0]:
                                           crop_corner[0] + crop_size])

    gaussian_images_sigmaY_crop = []
    for i in range(len(gaussian_images_sigmaY)):
        gaussian_images_sigmaY_crop.append(gaussian_images_sigmaY[i]
                                           [crop_corner[1]:
                                            crop_corner[1] + crop_size,
                                           crop_corner[0]:
                                           crop_corner[0] + crop_size])

    # Concat the arrays of sigmaX and sigmaY to plot
    plot_images = gaussian_images_sigmaX_crop + gaussian_images_sigmaY_crop
    # Concat the titles arrays of sigmaX and sigmaY to plot
    plot_titles = titles_images_X + titles_images_Y

    # Plots the images.
    fig = util.plotImages(plot_images,
                          plot_titles,
                          show=True,
                          main_title="Gaussian Filter Sigmas @ Kernel 9x9 - "
                                     "cv.GaussianBlur",
                          num=num,
                          dpi=dpi)

    # Saves the figure.
    if save != "None":
        fig.savefig(save)

    # Wait for a key press to close figures
    input("Press Enter to continue...")


if __name__ == "__main__":
    gaussian_filter_sigma()
