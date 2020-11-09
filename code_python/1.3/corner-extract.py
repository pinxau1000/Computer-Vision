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
_IMG_HARRIS_NAME = "Harris.jpg"
_FULL_PATH_ORIG = path.join(_PATH_2_DATA, _IMG_ORIG_NAME)
_FULL_PATH_NOISE = path.join(_PATH_2_DATA, _IMG_NOISE_NAME)
_FULL_PATH_HARRIS = path.join(_PATH_2_DATA, _IMG_HARRIS_NAME)


# ------------------------------------------------------------------------------
#                           Functions
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#                           Harris Corner Detector
# ------------------------------------------------------------------------------
# PASSED
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_ORIG,
              type=str,
              help="The path to the image")
@click.option("--bsize",
              default=3,
              type=int,
              help="Neighborhood size")
@click.option("--ksize",
              default=3,
              type=int,
              help="Aperture parameter for the Sobel operator")
@click.option("--k",
              default=0.06,
              type=float,
              help="Harris detector free parameter")
@click.option("--threshold",
              default=0.02,
              type=float,
              help="values above threshold*max(R) are considered corners.")
@click.option("--filter_params",
              default=str([0]),
              help="Tuple with initial filtering parameters. If None no "
                   "filter will be applied. If a 2 element tuple the a "
                   "Gaussian Blur will be applied with ksize=(filter_params["
                   "0], filter_params[1]).")
@click.option("--save",
              default="output_HarrisCornerDetector",
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
def harris_detector(image, bsize, ksize, k, threshold, filter_params, save,
                    dpi, num):
    image = util.load_image_RGB(image)
    filter_params = json.loads(filter_params)

    if len(filter_params) == 2:
        print("Applying Gaussian Filter")
        image = cv.GaussianBlur(src=image,
                                ksize=(filter_params[0], filter_params[1]),
                                sigmaX=0,
                                sigmaY=0)

    gray_image = cv.cvtColor(src=image, code=cv.COLOR_RGB2GRAY)

    harris_image = np.copy(image)
    mask = cv.cornerHarris(src=gray_image,
                           blockSize=bsize,
                           ksize=ksize,
                           k=k)

    harris_image[mask > threshold * np.max(mask)] = [255, 0, 0]

    # Convert from Float 64 to Unsigned Int 8
    # Also needs to be converted from np.array to list
    harris_image = list(np.uint8(np.abs(harris_image)))

    # Plots the images.
    fig = util.plotImages([image, harris_image],
                          ["Orig Image", "Harris Output"],
                          show=True,
                          main_title="Harris Corner Detector - cv.cornerHarris"
                                     f"\nblock size = {bsize}"
                                     f"\nsobel aperture = {ksize}"
                                     f"\nharris param = {k}",
                          cols=2,
                          num=num,
                          dpi=dpi)

    # Saves the figure.
    if save != "None":
        fig.savefig(save)

    # Wait for a key press to close figures
    input("Press Enter to continue...")


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
def harris_detector_bsize(image, bsizes, ksize, k, threshold, filter_params,
                          save, dpi, num):
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
                          cmap="gray",
                          cols=2,
                          num=num,
                          dpi=dpi)

    # Saves the figure.
    if save != "None":
        fig.savefig(save)

    # Wait for a key press to close figures
    input("Press Enter to continue...")


# ------------------------------------------------------------------------------
#                       Harris Corner Detector Sobel Aperture
# ------------------------------------------------------------------------------
# PASSED
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_ORIG,
              type=str,
              help="The path to the image")
@click.option("--bsize",
              default=3,
              type=int,
              help="Neighborhood size")
@click.option("--ksizes",
              default=str(list(range(3, 16, 6))),
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
              default="output_HarrisCornerDetector_SobelAperture",
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
def harris_detector_ksize(image, bsize, ksizes, k, threshold, filter_params,
                          save, dpi, num):
    image = util.load_image_RGB(image)
    ksizes = json.loads(ksizes)
    ksizes = np.rint(ksizes).astype(int)
    filter_params = json.loads(filter_params)

    if len(filter_params) == 2:
        print("Applying Gaussian Filter")
        image = cv.GaussianBlur(src=image,
                                ksize=(filter_params[0], filter_params[1]),
                                sigmaX=0,
                                sigmaY=0)

    gray_image = cv.cvtColor(src=image, code=cv.COLOR_RGB2GRAY)

    harris_images_ksize = []
    harris_images_titles = []
    for ksize in ksizes:
        harris_image = np.copy(image)
        mask = cv.cornerHarris(src=gray_image,
                               blockSize=bsize,
                               ksize=ksize,
                               k=k)

        harris_image[mask > threshold * mask.max()] = [255, 0, 0]
        harris_images_ksize.append(harris_image)

        harris_images_titles.append(f"sobel aperture = {ksize}")

    harris_images = harris_images_ksize
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
                          main_title="Harris Corner Detector - cv.cornerHarris"
                                     f"\nblock size = {bsize}"
                                     f"\nharris parameter = {k}",
                          cmap="gray",
                          cols=2,
                          num=num,
                          dpi=dpi)

    # Saves the figure.
    if save != "None":
        fig.savefig(save)

    # Wait for a key press to close figures
    input("Press Enter to continue...")


# ------------------------------------------------------------------------------
#                       Harris Corner Detector Harris Parameter
# ------------------------------------------------------------------------------
# PASSED
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_ORIG,
              type=str,
              help="The path to the image")
@click.option("--bsize",
              default=3,
              type=int,
              help="Neighborhood size")
@click.option("--ksize",
              default=5,
              type=int,
              help="Aperture parameter for the Sobel operator")
@click.option("--ks",
              default=str(list(np.array([-0.08, -0.02, 0, 0.02, 0.08]))),
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
              default="output_HarrisCornerDetector_HarrisParameter",
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
def harris_detector_k(image, bsize, ksize, ks, threshold, filter_params,
                      save, dpi, num):
    image = util.load_image_RGB(image)
    ks = json.loads(ks)
    ks = np.round(ks, 2)
    filter_params = json.loads(filter_params)

    if len(filter_params) == 2:
        print("Applying Gaussian Filter")
        image = cv.GaussianBlur(src=image,
                                ksize=(filter_params[0], filter_params[1]),
                                sigmaX=0,
                                sigmaY=0)

    gray_image = cv.cvtColor(src=image, code=cv.COLOR_RGB2GRAY)

    harris_images_ks = []
    harris_images_titles = []
    for k in ks:
        harris_image = np.copy(image)
        mask = cv.cornerHarris(src=gray_image,
                               blockSize=bsize,
                               ksize=ksize,
                               k=k)

        harris_image[mask > threshold * np.max(mask)] = [255, 0, 0]
        harris_images_ks.append(harris_image)

        harris_images_titles.append(f"harris parameter={k}")

    harris_images = harris_images_ks
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
                          main_title="Harris Corner Detector - cv.cornerHarris"
                                     f"\n block size = {bsize}"
                                     f"\nsobel aperture = {ksize}",
                          cmap="gray",
                          cols=3,
                          num=num,
                          dpi=dpi)

    # Saves the figure.
    if save != "None":
        fig.savefig(save)

    # Wait for a key press to close figures
    input("Press Enter to continue...")


# region
@click.group()
def entry_point():
    pass


entry_point.add_command(harris_detector)
entry_point.add_command(harris_detector_bsize)
entry_point.add_command(harris_detector_ksize)
entry_point.add_command(harris_detector_k)
entry_point.add_command(harris_detector_animate)

if __name__ == "__main__":
    entry_point()
# endregion
