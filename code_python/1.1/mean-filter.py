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
_IMG_NOISE_NAME = "img05_noise.jpg"
_FULL_PATH_NOISE = path.join(_PATH_2_DATA, _IMG_NOISE_NAME)


# ------------------------------------------------------------------------------
#                               Mean Filter
# ------------------------------------------------------------------------------
# PASSED
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_NOISE,
              type=str,
              help="The path to the image")
@click.option("--kernels",
              default=str(list(range(3, 8, 2))),
              help="List of kernel dimensions")
@click.option("--save",
              default="output_MeanFilter",
              type=str,
              help="The save name(s) of the output figure(s)")
def mean_filter(image, kernels, save):
    image = util.load_image_RGB(image)
    kernels = json.loads(kernels)

    # Initialize the mean_images as list. Values are assigned on the for-loop
    mean_images = []
    titles_images = []
    for k in kernels:
        mean_images.append(cv.blur(image, (k, k)))
        titles_images.append(f"Kernel {k}x{k}")

    # Copy the arrays with the images generated in the for-loop and adds the
    # original noisy image for comparison. Also copies and adds the titles.
    plot_images = mean_images
    plot_images.insert(0, image)
    plot_titles = titles_images
    plot_titles.insert(0, "Noisy Image")

    # Plots the images.
    fig = util.plotImages(plot_images,
                          plot_titles,
                          show=True,
                          main_title="Mean Filter - cv.blur",
                          num=200,
                          dpi=300)

    # Saves the figure.
    fig.savefig(save)


if __name__ == "__main__":
    mean_filter()