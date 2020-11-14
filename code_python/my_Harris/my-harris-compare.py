# ------------------------------------------------------------------------------
#                                   IMPORTS
# ------------------------------------------------------------------------------
from sys import version_info
from sys import path as syspath
from os import path

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
_IMG_HARRIS_NAME = "Harris.jpg"
_FULL_PATH_HARRIS = path.join(_PATH_2_DATA, _IMG_HARRIS_NAME)


# ------------------------------------------------------------------------------
#                           Functions
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
#                           My Harris Corner Detector
# ------------------------------------------------------------------------------
# http://www.kaij.org/blog/?p=89
# https://webnautes.tistory.com/1291
# http://fs2.american.edu/bxiao/www/CSC589/lecture15.pdf
# https://muthu.co/harris-corner-detector-implementation-in-python/
@click.command()
@click.option("--image", prompt="Path",
              default=_FULL_PATH_HARRIS,
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
@click.option("--k",
              default=0.04,
              type=float,
              help="Harris detector free parameter")
@click.option("--threshold",
              default=0.01,
              type=float,
              help="Values below -threshold*max(R) are considered edged. "
                   "values above threshold*max(R) are considered corners. "
                   "Values between these are considered flat areas.")
@click.option("--save",
              default="output_HarrisCornerDetector_Compare",
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
def my_harris_compare(image, bsize, ksize, k, threshold, save, dpi, num):
    assert type(bsize) is int, "blockSize must be type int"
    assert type(ksize) is int, "ksize must be type int"
    assert type(k) is float, "k must be type float"
    assert bsize > 0, "blockSize must be greater than 0"
    assert ksize == 1 or ksize == 3 or ksize == 5 or ksize == 7, \
        "ksize must be type 1, 3, 5, or 7."

    image = util.load_image_RGB(image)
    gray_image = cv.cvtColor(src=image, code=cv.COLOR_RGB2GRAY)

    # My Harris Corner Detector
    Ix = cv.Sobel(src=gray_image,
                  ddepth=cv.CV_64F,
                  dx=1,
                  dy=0,
                  ksize=ksize)
    Iy = cv.Sobel(src=gray_image,
                  ddepth=cv.CV_64F,
                  dx=0,
                  dy=1,
                  ksize=ksize)

    IxIx = Ix ** 2
    IyIy = Iy ** 2
    IxIy = Ix * Iy

    height, width = np.shape(gray_image)[:2]
    my_R = np.empty((height, width))

    if bsize == 1:
        detM = IxIx * IyIy - IxIy ** 2
        traceM = IxIx + IyIy
        my_R = detM - k * (traceM ** 2)
    else:
        offset = int(bsize / 2)
        for row in range(offset, height - offset):
            for col in range(offset, width - offset):
                Sum_IxIx = np.sum(IxIx[
                                  row - offset:row + offset + 1,
                                  col - offset:col + offset + 1])
                Sum_IyIy = np.sum(IyIy[
                                  row - offset:row + offset + 1,
                                  col - offset:col + offset + 1])
                Sum_IxIy = np.sum(IxIy[
                                  row - offset:row + offset + 1,
                                  col - offset:col + offset + 1])
                detM = Sum_IxIx * Sum_IyIy - Sum_IxIy ** 2
                traceM = Sum_IxIx + Sum_IyIy
                my_R[row, col] = detM - k * (traceM ** 2)

    my_harris_image = np.copy(image)
    my_harris_image[my_R >= threshold * np.max(my_R)] = [255, 0, 0]

    # OpenCV Harris Corner Detector
    cv_R = cv.cornerHarris(src=gray_image,
                           blockSize=bsize,
                           ksize=ksize,
                           k=k)

    cv_harris_image = np.copy(image)
    cv_harris_image[cv_R > threshold * np.max(cv_R)] = [255, 0, 0]

    # test
    # for _ in range(100):
    #     cv_harris_image[np.random.randint(0, np.shape(cv_harris_image)[0])] \
    #         [np.random.randint(0, np.shape(cv_harris_image)[1])] = \
    #         [np.random.randint(0, np.max(cv_harris_image)),
    #         np.random.randint(0, np.max(cv_harris_image)),
    #         np.random.randint(0, np.max(cv_harris_image))]
    # test end

    cv_harris_image = list(np.uint8(np.abs(cv_harris_image)))

    dif_image = np.uint8(np.abs(cv_harris_image - my_harris_image))
    _, dif_image = cv.threshold(dif_image, 1, 255, cv.CV_8U)

    fig = util.plotImages([image, cv_harris_image, my_harris_image, dif_image],
                          ["Orig Image", "OpenCV", "Custom", "Differences"],
                          main_title="Harris Algorithm Comparison",
                          show=True,
                          num=num,
                          dpi=dpi)

    # Saves the figure.
    if save != "None":
        fig.savefig(save)

    # Wait for a key press to close figures
    input("Press Enter to continue...")


if __name__ == "__main__":
    my_harris_compare()
