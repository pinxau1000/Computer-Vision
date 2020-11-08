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
#                           Functions
# ------------------------------------------------------------------------------

# region Sobel Filter
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


# endregion

# region Sobel Filter Ddepth Problems
# ------------------------------------------------------------------------------
#                           Sobel Filter Ddepth Problems
# ------------------------------------------------------------------------------
# PASSED
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_ORIG,
              type=str,
              help="The path to the image")
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
              default="output_SobelFilter_dataTypesProblems",
              type=str,
              help="The save name(s) of the output figure(s)")
def sobel_filter_ddepth(image, ksize, threshold, filter_params, save):
    image = util.load_image_RGB(image)
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

    image_gray = cv.cvtColor(src=image, code=cv.COLOR_RGB2GRAY)

    sobel_1st_float64 = cv.Sobel(src=image_gray,
                                 ddepth=cv.CV_64F,
                                 dx=1,
                                 dy=0,
                                 ksize=ksize)
    sobel_1st_float64 = np.uint8(np.abs(sobel_1st_float64))

    sobel_1st_uint8 = cv.Sobel(src=image_gray,
                               ddepth=cv.CV_8U,
                               dx=1,
                               dy=0,
                               ksize=ksize)

    sobel_2nd_float64 = cv.Sobel(src=image_gray,
                                 ddepth=cv.CV_64F,
                                 dx=2,
                                 dy=0,
                                 ksize=ksize)
    sobel_2nd_float64 = np.uint8(np.abs(sobel_2nd_float64))

    sobel_2nd_uint8 = cv.Sobel(src=image_gray,
                               ddepth=cv.CV_8U,
                               dx=2,
                               dy=0,
                               ksize=ksize)

    plot_images = [sobel_1st_float64, sobel_1st_uint8, sobel_2nd_float64,
                   sobel_2nd_uint8]
    plot_titles = ["dx=1 64F", "dx=1 8U", "dx=2 64F", "dx=2 8U"]

    # Plots the Black and White images.
    fig = util.plotImages(plot_images,
                          plot_titles,
                          show=True,
                          main_title="Sobel Derivatives Problems 1 - cv.Sobel",
                          num=202,
                          dpi=300,
                          cmap="gray")

    # Saves the figure.
    fig.savefig(save + "_1")

    image_1st_uint8 = np.copy(image)
    image_1st_float64 = np.copy(image)
    image_2nd_uint8 = np.copy(image)
    image_2nd_float64 = np.copy(image)

    image_1st_uint8[sobel_1st_uint8 > threshold * np.max(sobel_1st_uint8)] \
        = [255, 0, 0]
    image_1st_float64[sobel_1st_float64 > threshold * np.max(sobel_1st_float64)] \
        = [255, 0, 0]
    image_2nd_uint8[sobel_2nd_uint8 > threshold * np.max(sobel_2nd_uint8)] \
        = [255, 0, 0]
    image_2nd_float64[sobel_2nd_float64 > threshold * np.max(sobel_2nd_float64)] \
        = [255, 0, 0]

    plot_images = [image_1st_float64, image_1st_uint8,
                   image_2nd_float64, image_2nd_uint8]
    plot_titles = ["dx=1 64F", "dx=1 8U", "dx=2 64F", "dx=2 8U"]

    # Plots the images.
    fig = util.plotImages(plot_images,
                          plot_titles,
                          show=True,
                          main_title="Sobel Derivatives Problems 2 - cv.Sobel",
                          num=203,
                          dpi=300)

    # Saves the figure.
    fig.savefig(save + "_2")


# endregion

# region Scharr Filter
# ------------------------------------------------------------------------------
#                               Scharr Filter
# ------------------------------------------------------------------------------
# PASSED
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_ORIG,
              type=str,
              help="The path to the image")
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
              default="output_ScharrFilter",
              type=str,
              help="The save name(s) of the output figure(s)")
def scharr_filter(image, filter_params, save):
    image = util.load_image_RGB(image)
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

    scharr_x = cv.Scharr(src=image, ddepth=cv.CV_64F, dx=1, dy=0)
    scharr_y = cv.Scharr(src=image, ddepth=cv.CV_64F, dx=0, dy=1)
    scharr_xy = cv.magnitude(scharr_x, scharr_y)

    scharr_x = list(np.uint8(np.abs(scharr_x)))
    scharr_y = list(np.uint8(np.abs(scharr_y)))
    scharr_xy = list(np.uint8(np.abs(scharr_xy)))

    scharr_images = [image,
                     scharr_x,
                     scharr_y,
                     scharr_xy]
    titles_images = ["Original Image",
                     "Scharr dx=1",
                     "Scharr dy=1",
                     "Magnitude"]

    # Plots the images.
    fig = util.plotImages(scharr_images,
                          titles_images,
                          show=True,
                          main_title="Scharr Filter - cv.Scharr",
                          num=300,
                          dpi=300,
                          cmap="gray")

    # Saves the figure.
    fig.savefig(save)


# endregion

# region Scharr Filter Threshold
# ------------------------------------------------------------------------------
#                               Scharr Filter Threshold
# ------------------------------------------------------------------------------
# PASSED
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_ORIG,
              type=str,
              help="The path to the image")
@click.option("--threshold",
              default=0.25,
              type=float,
              help="Values below threshold*max(ScharrMagnitude) are set to 0.")
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
              default="output_ScharrFilter_Threshold",
              type=str,
              help="The save name(s) of the output figure(s)")
def scharr_filter_threshold(image, threshold, filter_params, save):
    image = util.load_image_RGB(image)
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

    scharr_x = cv.Scharr(src=image, ddepth=cv.CV_64F, dx=1, dy=0)
    scharr_y = cv.Scharr(src=image, ddepth=cv.CV_64F, dx=0, dy=1)
    scharr_xy = cv.magnitude(scharr_x, scharr_y)

    scharr_x = list(np.uint8(np.abs(scharr_x)))
    scharr_y = list(np.uint8(np.abs(scharr_y)))
    scharr_xy = list(np.uint8(np.abs(scharr_xy)))

    scharr_x_thresh = np.copy(scharr_x)
    scharr_y_thresh = np.copy(scharr_y)
    scharr_xy_thresh = np.copy(scharr_xy)

    scharr_x_thresh[scharr_x_thresh < threshold * np.max(scharr_x_thresh)] = 0
    scharr_y_thresh[scharr_y_thresh < threshold * np.max(scharr_y_thresh)] = 0
    scharr_xy_thresh[
        scharr_xy_thresh < threshold * np.max(scharr_xy_thresh)] = 0

    scharr_images_thresh = [image,
                            scharr_x_thresh,
                            scharr_y_thresh,
                            scharr_xy_thresh]

    titles_images = ["Original Image",
                     "Scharr dx=1",
                     "Scharr dy=1",
                     "Magnitude"]

    # Plots the images.
    fig = util.plotImages(scharr_images_thresh,
                          titles_images,
                          show=True,
                          main_title=f"Scharr Filter with Threshold "
                                     f"{threshold} - cv.Scharr",
                          num=301,
                          dpi=300,
                          cmap="gray")

    # Saves the figure.
    fig.savefig(save)


# endregion

# region Prewitt Filter
# ------------------------------------------------------------------------------
#                               Prewitt Filter
# ------------------------------------------------------------------------------
# PASSED
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_ORIG,
              type=str,
              help="The path to the image")
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
              default="output_PrewittFilter",
              type=str,
              help="The save name(s) of the output figure(s)")
def prewitt_filter(image, filter_params, save):
    image = util.load_image_RGB(image)
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

    kernel_x = util.makeFilter([1, 0, -1], [1, 1, 1])
    kernel_y = util.makeFilter([1, 1, 1], [1, 0, -1])

    prewitt_x = cv.filter2D(src=image, ddepth=cv.CV_64F, kernel=kernel_x)
    prewitt_y = cv.filter2D(src=image, ddepth=cv.CV_64F, kernel=kernel_y)
    prewitt_xy = cv.magnitude(prewitt_x, prewitt_y)

    prewitt_x = list(np.uint8(np.abs(prewitt_x)))
    prewitt_y = list(np.uint8(np.abs(prewitt_y)))
    prewitt_xy = list(np.uint8(np.abs(prewitt_xy)))

    prewitt_images = [image,
                      prewitt_x,
                      prewitt_y,
                      prewitt_xy]
    titles_images = ["Orig Image",
                     "Prewitt xx",
                     "Prewitt yy",
                     "Magnitude"]

    # Plots the images.
    fig = util.plotImages(prewitt_images,
                          titles_images,
                          show=True,
                          main_title="Prewitt Filter - cv.filter2D",
                          num=400,
                          dpi=300,
                          cmap="gray")

    # Saves the figure.
    fig.savefig(save)


# endregion

# region Roberts Filter
# ------------------------------------------------------------------------------
#                               Roberts Filter
# ------------------------------------------------------------------------------
# PASSED
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_ORIG,
              type=str,
              help="The path to the image")
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
              default="output_RobertsFilter",
              type=str,
              help="The save name(s) of the output figure(s)")
def roberts_filter(image, filter_params, save):
    image = util.load_image_RGB(image)
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

    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])

    roberts_x = cv.filter2D(src=image, ddepth=cv.CV_64F, kernel=kernel_x)
    roberts_y = cv.filter2D(src=image, ddepth=cv.CV_64F, kernel=kernel_y)
    roberts_xy = cv.magnitude(roberts_x, roberts_y)

    # Before plotting we need to type convert
    roberts_x = list(np.uint8(np.abs(roberts_x)))
    roberts_y = list(np.uint8(np.abs(roberts_y)))
    roberts_xy = list(np.uint8(np.abs(roberts_xy)))

    roberts_images = [image,
                      roberts_x,
                      roberts_y,
                      roberts_xy]
    titles_images = ["Orig Image",
                     "Roberts xx",
                     "Roberts yy",
                     "Magnitude"]

    # Plots the images.
    fig = util.plotImages(roberts_images,
                          titles_images,
                          show=True,
                          main_title="Roberts Filter - cv.filter2D",
                          num=500,
                          dpi=300,
                          cmap="gray")

    # Saves the figure.
    fig.savefig(save)


# endregion

# region Canny Filter
# ------------------------------------------------------------------------------
#                               Canny Filter
# ------------------------------------------------------------------------------
# PASSED
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_ORIG,
              type=str,
              help="The path to the image")
@click.option("--minval",
              default=0.15,
              type=float,
              help="Minimum threshold. Values below minVal*max(image) are "
                   "considered not edges.")
@click.option("--maxval",
              default=0.3,
              type=float,
              help="Maximum threshold. Values above maxVal*max(image) are "
                   "considered edges.")
@click.option("--ksize",
              default=None,
              type=int,
              help="Size of the extended Sobel kernel; it must be 1, 3, 5, "
                   "or 7.")
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
              default="output_CannyFilter",
              type=str,
              help="The save name(s) of the output figure(s)")
def canny_filter(image, minval, maxval, ksize, filter_params, save):
    image = util.load_image_RGB(image)
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

    minVal = np.uint8(np.round(np.max(image) * minval))
    maxVal = np.uint8(np.round(np.max(image) * maxval))

    # Aperture size should be odd between 3 and 7 in function
    canny_image = cv.Canny(image=image,
                           threshold1=minVal,
                           threshold2=maxVal,
                           apertureSize=ksize)

    canny_images = [image, canny_image]
    titles_images = ["Orig Image",
                     f"Canny\nminVal = {minVal}, maxVal = {maxVal}"]

    # Plots the images.
    fig = util.plotImages(canny_images,
                          titles_images,
                          show=True,
                          main_title="Canny Filter - cv.Canny",
                          num=600,
                          dpi=300,
                          cmap="gray",
                          cols=2)

    # Saves the figure.
    fig.savefig(save)


# endregion

# region Canny Filter Animation
# ------------------------------------------------------------------------------
#                               Canny Filter Animation
# ------------------------------------------------------------------------------
# PASSED
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_ORIG,
              type=str,
              help="The path to the image")
@click.option("--minvals",
              default=str(list(np.linspace(0, 1, 25))),
              help="List of minimums threshold to animate. Values below "
                   "minVal*max(image) are considered not edges.")
@click.option("--maxvals",
              default=str(list(np.linspace(0, 1, 25))),
              help="List of maximums threshold to animate. Values above "
                   "maxVal*max(image) are considered edges.")
@click.option("--ksize",
              default=None,
              type=int,
              help="Size of the extended Sobel kernel; it must be 1, 3, 5, "
                   "or 7.")
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
              default="output_CannyEdgeDetector_Animation",
              type=str,
              help="The save name(s) of the output figure(s)")
def canny_filter_animate(image, minvals, maxvals, ksize, filter_params, save):
    image = util.load_image_RGB(image)
    minvals = json.loads(minvals)
    maxvals = json.loads(maxvals)
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

    # Aperture size should be odd between 3 and 7 in function
    canny_images = []
    canny_titles = []

    minvals = np.rint(np.max(image) * np.array(minvals)).astype(int)
    maxvals = np.rint(np.max(image) * np.array(maxvals)).astype(int)

    for thrsh1 in minvals:
        for thrsh2 in maxvals:
            canny_images.append(cv.Canny(image=image,
                                         threshold1=thrsh1,
                                         threshold2=thrsh2,
                                         apertureSize=ksize))
            canny_titles.append(f"minVal = {np.round(thrsh1)}, "
                                f"maxVal = {np.round(thrsh2)}")
        # Do not repeat values cause thrsh1 = 200 and thrsh2 = 180 is the same
        # as thrsh1 = 180 and thrsh2 = 200
        maxvals = np.delete(maxvals, np.argwhere(maxvals == thrsh1))

    util.animateImages(images=canny_images,
                       titles=canny_titles,
                       save_name=save,
                       cmap="gray",
                       frame_interval=120,
                       verbose=True)


# endregion

# region Laplacian Filter
# ------------------------------------------------------------------------------
#                               Laplacian Filter
# ------------------------------------------------------------------------------
# PASSED
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_ORIG,
              type=str,
              help="The path to the image")
@click.option("--ksize",
              default=None,
              type=int,
              help="Size of the extended Sobel kernel; it must be 1, 3, 5, "
                   "or 7.")
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
              default="output_LaplacianFilter",
              type=str,
              help="The save name(s) of the output figure(s)")
def laplacian_filter(image, ksize, filter_params, save):
    image = util.load_image_RGB(image)
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

    laplacian_image = cv.Laplacian(src=image, ddepth=cv.CV_64F, ksize=ksize)

    laplacian_image = np.uint8(np.abs(laplacian_image))

    laplacian_images = [image, laplacian_image]
    titles_images = ["Orig Image", "Laplacian"]

    # Plots the images.
    fig = util.plotImages(laplacian_images,
                          titles_images,
                          show=True,
                          main_title="Laplacian Filter - cv.Laplacian",
                          num=700,
                          dpi=300,
                          cmap="gray",
                          cols=2)

    # Saves the figure.
    fig.savefig(save)

# endregion
