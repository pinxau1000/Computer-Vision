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


if __name__ == "__main__":
    canny_filter_animate()
