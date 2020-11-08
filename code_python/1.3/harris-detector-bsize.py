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
_FULL_PATH_ORIG = path.join(_PATH_2_DATA, _IMG_ORIG_NAME)


# ------------------------------------------------------------------------------
#                       Harris Corner Detector Block Size
# ------------------------------------------------------------------------------
# PASSED
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_ORIG,
              type=str,
              help="The path to the image")
@click.option("--bsizes",
              default=str(list(range(3, 16, 6))),
              help="Neighborhood size")
@click.option("--ksize",
              default=5,
              type=int,
              help="Aperture parameter for the Sobel operator")
@click.option("--k",
              default=0.04,
              type=float,
              help="Harris detector free parameter")
@click.option("--threshold",
              default=0.01,
              type=float,
              help="values above threshold*max(R) are considered corners.")
@click.option("--filter_params",
              default=str([0]),
              help="Tuple with initial filtering parameters. If None no "
                   "filter will be applied. If a 2 element tuple the a "
                   "Gaussian Blur will be applied with ksize=(filter_params["
                   "0], filter_params[1]).")
@click.option("--save",
              default="output_HarrisCornerDetector_BlockSize",
              type=str,
              help="The save name(s) of the output figure(s)")
def harris_detector_bsize(image, bsizes, ksize, k, threshold, filter_params,
                          save):
    image = util.load_image_RGB(image)
    bsizes = json.loads(bsizes)
    bsizes = np.rint(bsizes).astype(int)
    filter_params = json.loads(filter_params)

    if len(filter_params) == 2:
        print("Applying Gaussian Filter")
        image = cv.GaussianBlur(src=image,
                                ksize=(filter_params[0], filter_params[1]),
                                sigmaX=0,
                                sigmaY=0)

    gray_image = cv.cvtColor(src=image, code=cv.COLOR_RGB2GRAY)

    harris_images_bsize = []
    harris_images_titles = []
    for bsize in bsizes:
        harris_image = np.copy(image)
        mask = cv.cornerHarris(src=gray_image,
                               blockSize=bsize,
                               ksize=ksize,
                               k=k)

        harris_image[mask > threshold * mask.max()] = [255, 0, 0]
        harris_images_bsize.append(harris_image)

        harris_images_titles.append(f"block size = {bsize}")

    harris_images = harris_images_bsize
    harris_images.insert(0, image)
    titles_images = harris_images_titles
    titles_images.insert(0, "Orig Image")

    # Convert from Float 64 to Unsigned Int 8
    # Also needs to be converted from np.array to list
    harris_images = list(np.uint8(np.abs(harris_images)))

    # Plots the images.
    fig = util.plotImages(harris_images,
                          titles_images,
                          show=True,
                          main_title="Harris Corner Detectorn - cv.cornerHarris"
                                     f"\nsobel aperture = {ksize}"
                                     f"\nharris parameter = {k}",
                          num=201,
                          dpi=300,
                          cmap="gray",
                          cols=2)

    # Saves the figure.
    fig.savefig(save)


if __name__ == "__main__":
    harris_detector_bsize()
