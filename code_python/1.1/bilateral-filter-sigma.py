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
#                               Bilateral Filter Sigma
# ------------------------------------------------------------------------------
# PASSED
# See https://bit.ly/35X9VhK for recommended values.
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_NOISE,
              type=str,
              help="The path to the image")
@click.option("--diameter",
              default=9,
              type=int,
              help="Diameter of each pixel neighborhood used during "
                   "filtering. If it is non-positive it is computed from "
                   "sigmaSpace.")
@click.option("--sigma_c",
              default=str(list(range(10, 251, 120))),
              help="Filter sigma in the color space. A larger value of the "
                   "parameter means that farther colors within the pixel "
                   "neighborhood (see sigmaSpace) will be mixed together, "
                   "resulting in larger areas of semi-equal color.")
@click.option("--sigma_s",
              default=str(list(range(3, 16, 6))),
              help="Filter sigma in the coordinate space. A larger value of "
                   "the parameter means that farther pixels will influence "
                   "each other as long as their colors are close enough (see "
                   "sigmaColor). When d>0, it specifies the neighborhood "
                   "size regardless of sigmaSpace.Otherwise, "
                   "d is proportional to sigmaSpace.")
@click.option("--crop_corner",
              default=str([480, 110]),
              help="The upper left corner of the crop area as list. The point "
                   "with coordinates x=480 and y=110 is passed as [480,110].")
@click.option("--crop_size",
              default=64,
              type=int,
              help="The size of the crop area")
@click.option("--save",
              default="output_BilateralFilter_Sigma",
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
def bilateral_filter_sigma(image, diameter, sigma_c, sigma_s, crop_corner,
                           crop_size, save, dpi, num):
    image = util.load_image_RGB(image)
    sigma_c = json.loads(sigma_c)
    sigma_s = json.loads(sigma_s)
    crop_corner = json.loads(crop_corner)

    # Initialize the bilateral_images as list.
    # Values are assigned on the for-loop
    bilateral_images_sigmaC = []
    titles_images_sigmaC = []
    _sigS = min(sigma_s)
    for sigC in sigma_c:
        bilateral_images_sigmaC.append(cv.bilateralFilter(src=image,
                                                          d=diameter,
                                                          sigmaColor=sigC,
                                                          sigmaSpace=_sigS))
        titles_images_sigmaC.append(f"SigmaC = {sigC}")

    bilateral_images_sigmaS = []
    titles_images_sigmaS = []
    _sigC = max(sigma_c)
    for sigS in sigma_s:
        bilateral_images_sigmaS.append(cv.bilateralFilter(src=image,
                                                          d=diameter,
                                                          sigmaColor=_sigC,
                                                          sigmaSpace=sigS))
        titles_images_sigmaS.append(f"SigmaS = {sigS}")

    # Crop filtered images to better see the effect of Sigma.
    bilateral_images_sigmaC_crop = []
    for i in range(len(bilateral_images_sigmaC)):
        bilateral_images_sigmaC_crop.append(bilateral_images_sigmaC[i]
                                            [crop_corner[1]:
                                             crop_corner[1] + crop_size,
                                            crop_corner[0]:
                                            crop_corner[0] + crop_size])

    bilateral_images_sigmaS_crop = []
    for i in range(len(bilateral_images_sigmaC)):
        bilateral_images_sigmaS_crop.append(bilateral_images_sigmaS[i]
                                            [crop_corner[1]:
                                             crop_corner[1] + crop_size,
                                            crop_corner[0]:
                                            crop_corner[0] + crop_size])

    plot_images = bilateral_images_sigmaC_crop + bilateral_images_sigmaS_crop
    plot_titles = titles_images_sigmaC + titles_images_sigmaS

    # Plots the images.
    fig = util.plotImages(plot_images,
                          plot_titles,
                          show=True,
                          main_title=f"Bilateral Filter Sigma @"
                                     f"D = {diameter} - cv.bilateralFilter",
                          num=num,
                          dpi=dpi)

    # Saves the figure.
    if save != "None":
        fig.savefig(save)

    # Wait for a key press to close figures
    input("Press Enter to continue...")


if __name__ == "__main__":
    bilateral_filter_sigma()
