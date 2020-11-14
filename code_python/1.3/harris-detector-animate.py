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
    util.install("numpy>=1.19,<1.19.4")
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
       version.parse(np.__version__).minor >= 19 and \
       version.parse(np.__version__).micro < 4, \
    "This script requires Numpy version >= 1.19.0 and < 1.19.4 !"

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
#                       Harris Corner Detector Animation
# ------------------------------------------------------------------------------
# PASSED
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_ORIG,
              type=str,
              help="The path to the image")
@click.option("--bsizes",
              default=str(list(range(1, 16, 2))),
              help="Neighborhood size")
@click.option("--ksizes",
              default=str(list(range(3, 16, 2))),
              help="Aperture parameter for the Sobel operator")
@click.option("--ks",
              default=str(list(np.array([-0.08, -0.04, -0.02, 0,
                                         0.02, 0.04, 0.08]))),
              help="Harris detector free parameter")
@click.option("--threshold",
              default=0.01,
              type=float,
              help="Values below threshold*max(SobelMagnitude) are set to 0.")
@click.option("--save",
              default="output_HarrisCornerDetector_Animation",
              type=str,
              help="The save name(s) of the output figure(s)")
def harris_detector_animate(image, bsizes, ksizes, ks, threshold, save):
    image = util.load_image_RGB(image)
    gray_image = cv.cvtColor(src=image, code=cv.COLOR_RGB2GRAY)

    bsizes = json.loads(bsizes)
    ksizes = json.loads(ksizes)
    ks = json.loads(ks)

    bsizes = np.rint(bsizes).astype(int)
    ksizes = np.rint(ksizes).astype(int)
    ks = np.round(ks, 2)

    harris_images = []
    harris_titles = []
    for bsize in bsizes:
        for ksize in ksizes:
            for k in ks:
                # When the number its even add one to ensure ksize is odd
                ksize = ksize + 1 if ksize % 2 == 0 else ksize
                harris_image = np.copy(image)
                mask = cv.cornerHarris(src=gray_image,
                                       blockSize=bsize,
                                       ksize=ksize,
                                       k=k)

                harris_image[mask > threshold * mask.max()] = [255, 0, 0]
                harris_images.append(harris_image)

                harris_titles.append(f"bsize={bsize}, ksize={ksize}, k={k}")

    # Convert from Float 64 to Unsigned Int 8
    # Also needs to be converted from np.array to list
    harris_images = list(np.uint8(np.abs(harris_images)))

    util.animateImages(images=harris_images,
                       titles=harris_titles,
                       save_name=save,
                       cmap="gray",
                       frame_interval=120,
                       verbose=True)


if __name__ == "__main__":
    harris_detector_animate()
