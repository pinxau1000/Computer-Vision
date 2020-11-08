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
#                                   Sobel Filter
# ------------------------------------------------------------------------------
# PASSED
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_ORIG,
              type=str,
              help="The path to the image")
@click.option("--deriv_x",
              default=str(list(range(1, 3))),
              help="List with X derivatives order")
@click.option("--deriv_y",
              default=str(list(range(1, 3))),
              help="List with Y derivatives order")
@click.option("--ksize",
              default=None,
              type=int,
              help="Size of the extended Sobel kernel; it must be 1, 3, 5, "
                   "or 7.")
@click.option("--threshold",
              default=0.125,
              type=float,
              help="Values below threshold*max(SobelMagnitude) are set to 0.")
@click.option("--filter_params",
              default=str([0]),
              help="Tuple with initial filtering parameters. If None no "
                   "filter will be applied. If a 2 element tuple the a "
                   "Gaussian Blur will be applied with ksize=(filter_params["
                   "0], filter_params[1]). If a 3 elements tuple a bilateral "
                   "filtering will be applied with d=filter_params[0], "
                   "sigmaColor=filter_params[1] and sigmaSpace=filter_params["
                   "2]")
@click.option("--save",
              default="output_SobelFilter",
              type=str,
              help="The save name(s) of the output figure(s)")
def sobel_filter(image, deriv_x, deriv_y, ksize, threshold,
                 filter_params, save):
    image = util.load_image_RGB(image)
    deriv_x = json.loads(deriv_x)
    deriv_y = json.loads(deriv_y)
    filter_params = json.loads(filter_params)

    if len(filter_params) == 2:
        print("Applying Gaussian Filter")
        image = cv.GaussianBlur(src=image,
                                ksize=(filter_params[0], filter_params[1]),
                                sigmaX=0,
                                sigmaY=0)
    # Best bilateral_params = (6, 200, 20)
    elif len(filter_params) == 3:
        print("Applying Bilateral Filter")
        image = cv.bilateralFilter(src=image,
                                   d=filter_params[0],
                                   sigmaColor=filter_params[1],
                                   sigmaSpace=filter_params[2])

    image = cv.cvtColor(src=image, code=cv.COLOR_RGB2GRAY)

    # Initialize the mean_images as list. Values are assigned on the for-loop
    sobel_images_dx = []
    titles_images_dx = []
    for dx in deriv_x:
        # If not specified or set to None KSize is 3!
        sobel_images_dx.append(cv.Sobel(src=image,
                                        ddepth=cv.CV_64F,
                                        dx=dx,
                                        dy=0,
                                        ksize=ksize))
        titles_images_dx.append(f"dx = {dx}")

    sobel_images_dy = []
    titles_images_dy = []
    for dy in deriv_y:
        sobel_images_dy.append(cv.Sobel(src=image,
                                        ddepth=cv.CV_64F,
                                        dx=0,
                                        dy=dy,
                                        ksize=ksize))
        titles_images_dy.append(f"dy = {dy}")

    mag = cv.magnitude(sobel_images_dx[-1], sobel_images_dy[-1])
    ang = cv.phase(sobel_images_dx[-1], sobel_images_dy[-1],
                   angleInDegrees=True)

    # Values below threshold are set to zero.
    # Needed to visualize the orientation/angle
    _, mask = cv.threshold(mag, np.max(mag) * threshold, 1, cv.THRESH_BINARY)

    mag = np.multiply(mask, mag)
    ang = np.multiply(mask, ang)

    sobel_images_dxdy = [mag, ang]

    titles_images_dxdy = [f"Magnitude\ndx = {max(deriv_x)}, "
                          f"dy = {max(deriv_y)}",
                          f"Orientation\ndx = {max(deriv_x)}, "
                          f"dy = {max(deriv_y)}"]

    # Before plotting we need to type convert
    sobel_images_dx = list(np.uint8(np.abs(sobel_images_dx)))
    sobel_images_dy = list(np.uint8(np.abs(sobel_images_dy)))
    sobel_images_dxdy = list(np.uint8(np.abs(sobel_images_dxdy)))

    # Copy the arrays with the images generated in the for-loop and adds the
    # original noisy image for comparison. Also copies and adds the titles.
    plot_images = sobel_images_dx + sobel_images_dy + [sobel_images_dxdy[0]]
    plot_images.insert(0, image)

    plot_titles = titles_images_dx + titles_images_dy + [titles_images_dxdy[0]]
    plot_titles.insert(0, "Original Image")

    # Plots the images.
    fig = util.plotImages(plot_images,
                          plot_titles,
                          show=True,
                          main_title="Sobel Filter - cv.Sobel",
                          num=200,
                          dpi=300,
                          cmap="gray")

    # Saves the figure.
    fig.savefig(save)

    # Plots the images.
    fig = util.plotImages(sobel_images_dxdy,
                          titles_images_dxdy,
                          show=True,
                          main_title="Sobel Filter - cv.Sobel",
                          cols=2,
                          num=201,
                          dpi=300,
                          cmap="turbo")

    # Saves the figure.
    fig.savefig(save + "_MagAng")


if __name__ == "__main__":
    sobel_filter()
