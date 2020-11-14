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
_IMG_NOISE_NAME = "img05_noise.jpg"
_IMG_HARRIS_NAME = "Harris.jpg"
_FULL_PATH_ORIG = path.join(_PATH_2_DATA, _IMG_ORIG_NAME)
_FULL_PATH_NOISE = path.join(_PATH_2_DATA, _IMG_NOISE_NAME)
_FULL_PATH_HARRIS = path.join(_PATH_2_DATA, _IMG_HARRIS_NAME)


# ------------------------------------------------------------------------------
#                           Functions
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#                           Plot Original Pictures
# ------------------------------------------------------------------------------
# PASSED
@click.command()
@click.option('--orig',
              default=_FULL_PATH_ORIG,
              type=str,
              help="The path to the original image")
@click.option('--noisy',
              default=_FULL_PATH_NOISE,
              type=str,
              help='he path to the original image w/ noise')
@click.option("--save",
              default="output_OriginalPictures",
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
def original_pictures(orig, noisy, save, dpi, num):
    orig = util.load_image_RGB(orig)
    noisy = util.load_image_RGB(noisy)

    fig = util.plotImages([orig, noisy],
                          ["Original", "Noisy"],
                          show=True,
                          main_title="Loaded Images",
                          cols=2,
                          num=num,
                          dpi=dpi)

    # Saves the figure.
    if save != "None":
        fig.savefig(save)

    # Wait for a key press to close figures
    input("Press Enter to continue...")


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
              help="The save name(s) of the output figure(s). If None didn't "
                   "save the figure window.")
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
def mean_filter(image, kernels, save, dpi, num):
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
                          num=num,
                          dpi=dpi)

    # Saves the figure.
    if save != "None":
        fig.savefig(save)

    # Wait for a key press to close figures
    input("Press Enter to continue...")


# ------------------------------------------------------------------------------
#                               Mean Filter Anchor
# ------------------------------------------------------------------------------
# PASSED
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_NOISE,
              type=str,
              help="The path to the image")
@click.option("--kernel",
              default=10,
              help="Kernel aperture")
@click.option("--crop_corner",
              default=str([480, 110]),
              help="The upper left corner of the crop area as list. The point "
                   "with coordinates x=480 and y=110 is passed as [480,110].")
@click.option("--crop_size",
              default=64,
              type=int,
              help="The size of the crop area")
@click.option("--save",
              default="output_MeanFilter_Anchor",
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
def mean_filter_anchor(image, kernel, crop_corner, crop_size, save, dpi, num):
    image = util.load_image_RGB(image)
    crop_corner = json.loads(crop_corner)

    # Initialize the anchor_images as list. Values are assigned on the for-loop
    # Initializes the kernel_max which is the maximum kernel value from Mean
    # Filter section above.
    anchor_images = []
    titles_images = []

    for a in range(0, kernel, round((kernel - 1) / 2) - 1):
        anchor_images.append(cv.blur(image, (kernel, kernel),
                                     anchor=(a, a)))
        titles_images.append(f"Anchor at ({a}, {a})")

    # Crop blurred images to better see the effect of anchor.
    anchor_images_crop = []
    for i in range(len(anchor_images)):
        anchor_images_crop.append(anchor_images[i]
                                  [crop_corner[1]:crop_corner[1] + crop_size,
                                  crop_corner[0]:crop_corner[0] + crop_size])

    # Saves individual the images.
    # util.saveImages(anchor_images_crop, titles_images, dpi=300,
    #                 save_name=save)

    # Saves an animation.
    util.animateImages(images=anchor_images_crop,
                       titles=titles_images,
                       save_name=save,
                       frame_interval=120,
                       verbose=True)

    # Plots the images.
    fig = util.plotImages(anchor_images_crop,
                          titles_images,
                          show=True,
                          main_title=f"Mean Filter Anchor @ Kernel "
                                     f"{kernel}x{kernel} - cv.blur",
                          num=num,
                          dpi=dpi)

    # Saves the figure.
    if save != "None":
        fig.savefig(save)

    # Wait for a key press to close figures
    input("Press Enter to continue...")


# ------------------------------------------------------------------------------
#                               Median filter
# ------------------------------------------------------------------------------
# PASSED
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_NOISE,
              type=str,
              help="The path to the image")
@click.option("--kernels",
              default=str(list(range(5, 10, 2))),
              help="List with kernel dimensions")
@click.option("--save",
              default="output_MedianFilter",
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
def median_filter(image, kernels, save, dpi, num):
    image = util.load_image_RGB(image)
    kernels = json.loads(kernels)

    # Initialize the median_images as list. Values are assigned on the for-loop
    median_images = []
    titles_images = []
    for k in kernels:
        median_images.append(cv.medianBlur(image, k))
        titles_images.append(f"Kernel {k}x{k}")

    # Copy the arrays with the images generated in the for-loop and adds the
    # original noisy image for comparison. Also copies and adds the titles.
    plot_images = median_images
    plot_images.insert(0, image)
    plot_titles = titles_images
    plot_titles.insert(0, "Noisy Image")

    # Plots the images.
    fig = util.plotImages(plot_images,
                          plot_titles,
                          show=True,
                          main_title="Median Filter - cv.medianBlur",
                          num=num,
                          dpi=dpi)

    # Saves the figure.
    if save != "None":
        fig.savefig(save)

    # Wait for a key press to close figures
    input("Press Enter to continue...")


# ------------------------------------------------------------------------------
#                               Gaussian Filter
# ------------------------------------------------------------------------------
# PASSED
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_NOISE,
              type=str,
              help="The path to the image")
@click.option("--kernels",
              default=str(list(range(5, 10, 2))),
              help="List with kernel dimensions")
@click.option("--save",
              default="output_GaussianFilter",
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
def gaussian_filter(image, kernels, save, dpi, num):
    image = util.load_image_RGB(image)
    kernels = json.loads(kernels)

    # Initialize the gaussian_images as list.
    # Values are assigned on the for-loop
    gaussian_images = []
    titles_images = []
    for k in kernels:
        # SigmaX and SigmaY is 0 so they are calculated from kernel
        gaussian_images.append(cv.GaussianBlur(image, (k, k), 0, 0))
        titles_images.append(f"Kernel {k}x{k}")

    # Copy the arrays with the images generated in the for-loop and adds the
    # original noisy image for comparison. Also copies and adds the titles.
    plot_images = gaussian_images
    plot_images.insert(0, image)
    plot_titles = titles_images
    plot_titles.insert(0, "Noisy Image")

    # Plots the images.
    fig = util.plotImages(plot_images,
                          plot_titles,
                          show=True,
                          main_title="Gaussian Filter - cv.GaussianBlur",
                          num=num,
                          dpi=dpi)

    # Saves the figure.
    if save != "None":
        fig.savefig(save)

    # Wait for a key press to close figures
    input("Press Enter to continue...")


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


# ------------------------------------------------------------------------------
#                               Bilateral Filter
# ------------------------------------------------------------------------------
# PASSED
# See https://bit.ly/35X9VhK for recommended values.
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_NOISE,
              type=str,
              help="The path to the image")
@click.option("--diameters",
              default=str(list(range(5, 16, 5))),
              help="Diameter of each pixel neighborhood used during "
                   "filtering. If it is non-positive it is computed from "
                   "sigmaSpace.")
@click.option("--sigma_c",
              default=80,
              type=int,
              help="Filter sigma in the color space. A larger value of the "
                   "parameter means that farther colors within the pixel "
                   "neighborhood (see sigmaSpace) will be mixed together, "
                   "resulting in larger areas of semi-equal color.")
@click.option("--sigma_s",
              default=80,
              type=int,
              help="Filter sigma in the coordinate space. A larger value of "
                   "the parameter means that farther pixels will influence "
                   "each other as long as their colors are close enough (see "
                   "sigmaColor). When d>0, it specifies the neighborhood "
                   "size regardless of sigmaSpace.Otherwise, "
                   "d is proportional to sigmaSpace.")
@click.option("--save",
              default="output_BilateralFilter",
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
def bilateral_filter(image, diameters, sigma_c, sigma_s, save, dpi, num):
    image = util.load_image_RGB(image)
    diameters = json.loads(diameters)

    # Initialize the bilateral_images as list.
    # Values are assigned on the for-loop
    bilateral_images = []
    titles_images = []
    for d in diameters:
        bilateral_images.append(cv.bilateralFilter(src=image,
                                                   d=d,
                                                   sigmaColor=sigma_c,
                                                   sigmaSpace=sigma_s))
        titles_images.append(f"D = {d}")

    # Copy the arrays with the images generated in the for-loop and adds the
    # original noisy image for comparison. Also copies and adds the titles.
    plot_images = bilateral_images
    plot_images.insert(0, image)
    plot_titles = titles_images
    plot_titles.insert(0, "Noisy Image")

    # Plots the images.
    fig = util.plotImages(plot_images,
                          plot_titles,
                          show=True,
                          main_title=f"Bilateral Filter @ sigmaC = {sigma_c}, "
                                     f"sigmaS = {sigma_s} - cv.bilateralFilter",
                          num=num,
                          dpi=dpi)

    # Saves the figure.
    if save != "None":
        fig.savefig(save)

    # Wait for a key press to close figures
    input("Press Enter to continue...")


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


# region
@click.group()
def entry_point():
    pass


entry_point.add_command(original_pictures)
entry_point.add_command(mean_filter)
entry_point.add_command(mean_filter_anchor)
entry_point.add_command(median_filter)
entry_point.add_command(gaussian_filter)
entry_point.add_command(gaussian_filter_sigma)
entry_point.add_command(bilateral_filter)
entry_point.add_command(bilateral_filter_sigma)

if __name__ == "__main__":
    entry_point()
# endregion
