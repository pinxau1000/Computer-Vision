from typing import Union
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap

def plotImages(images: list, titles: list = None, num: int = None,
               show: bool = False, rows: int = None, cols: int = None,
               main_title: str = None, dpi: int = None,
               cmap: Union[str, Colormap] = 'viridis') -> plt.figure:
    assert type(images) is list, "Images should be list or dict."
    assert len(images) != 0, "Images is empty."

    from math import floor
    from math import ceil

    n_img = len(images)

    if cols is None:
        cols = floor(n_img / 2)

    if rows is None:
        if cols == 0:
            cols = 1
        rows = ceil(n_img / cols)

    _fig = plt.figure(num=num, dpi=dpi, clear=True, )
    for i in range(n_img):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i])
        plt.xticks([]), plt.yticks([])
        plt.set_cmap(cmap)
        try:
            plt.title(titles[i])
        except IndexError:
            pass
        except TypeError:
            pass

    if main_title:
        _fig.suptitle(main_title)

    if show:
        _fig.show()

    return _fig


def animateImages(images: list,
                  titles: list = None,
                  dpi: int = None,
                  save_name: str = "_temp",
                  frame_interval: int = 60,
                  repeat_delay: int = 1000,
                  cmap: Union[str, Colormap] = 'viridis',
                  verbose: bool = False):

    assert type(images) is list, "Images should be list or dict."
    assert len(images) != 0, "Images is empty."

    from matplotlib import animation
    from sys import stdout as stdout

    _str_stdout = ""
    if verbose:
        _str_stdout = "Compiling Figures: 0 % / 100 %"
        stdout.write(_str_stdout)

    _frames = []
    n_img = len(images)

    _fig = plt.figure(dpi=dpi)
    for i in range(n_img):
        plt.imshow(images[i], animated=True)
        plt.xticks([]), plt.yticks([])
        plt.set_cmap(cmap)
        title = ""
        try:
            title = plt.text(0.5, 1.01, titles[i],
                             horizontalalignment='center',
                             verticalalignment='bottom',
                             transform=plt.gca().transAxes)
        except IndexError:
            pass
        _frames.append([plt.gci(), title])
        if verbose:
            for _ in _str_stdout:
                stdout.write("\b")
            _str_stdout = f"Compiling figures: " \
                          f"{int(np.round(i*100/n_img))} % / 100 %"
            stdout.write(_str_stdout)

    stdout.write("\nCreating animation object, this may take a while.")
    ani = animation.ArtistAnimation(_fig,
                                    _frames,
                                    interval=frame_interval,
                                    blit=True,
                                    repeat_delay=repeat_delay)

    stdout.write(f"\nSaving to figures {save_name}, this may take a while.")
    try:
        ani.save(f'{save_name}.mp4')
    except ValueError:
        ani.save(f'{save_name}.gif')

    return


def saveImages(images: list,
               titles: list = None,
               dpi: int = None,
               save_name: str = "_img",
               cmap: Union[str, Colormap] = 'viridis'):
    assert type(images) is list, "Images should be list or dict."
    assert len(images) != 0, "Images is empty."

    files = []
    _fname = ""
    n_img = len(images)

    plt.figure(dpi=dpi)
    for i in range(n_img):
        plt.imshow(images[i])
        plt.xticks([]), plt.yticks([])
        plt.set_cmap(cmap)
        try:
            plt.title(titles[i])
        except IndexError:
            pass

        _fname = f"{save_name}_{i}.png"
        plt.savefig(_fname)
        files.append(_fname)

    return


def install(package):
    from sys import executable
    from subprocess import check_call

    check_call([executable, "-m", "pip", "install", package])


def checkPackage(packages: list = None) -> list:
    assert type(packages) is list, \
        "packages must be a list with required packages names or a list of " \
        "tuples with required packages names and required version!"

    from sys import executable
    import subprocess
    import json

    out_ = subprocess.run([executable, "-m", "pip", "list", "--format=json"],
                          capture_output=True)

    installed_packages = json.loads(out_.stdout)

    for _pckg in installed_packages:
        if _pckg["name"] in packages:
            packages.remove(_pckg["name"])

    # Return packages that where not found
    return packages


def checkPackageAndVersion(packages: list = None) -> list:
    assert type(packages) is list, \
        "packages must be a list with required packages names or a list of " \
        "tuples with required packages names and required version!"

    from sys import executable
    import subprocess
    import json

    out_ = subprocess.run([executable, "-m", "pip", "list", "--format=json"],
                          capture_output=True)

    installed_packages = json.loads(out_.stdout)
    missing_packages = packages.copy()

    for _mpckg in packages:
        for _ipckg in installed_packages:
            if type(_mpckg) is tuple:
                if _mpckg[0] == _ipckg["name"]:
                    # print(f">{_mpckg[0]}:\t"
                    #      f"req: {_mpckg[1]}\t"
                    #      f"inst: {_ipckg['version']}")
                    if _mpckg[1] == _ipckg["version"]:
                        missing_packages.remove(_mpckg)
                    else:
                        _mpckg_splt = _mpckg[1].split('.')
                        _ipckg_splt = _ipckg["version"].split('.')
                        _flag_installed = True
                        for i in range(len(_mpckg_splt)):
                            if _mpckg_splt[i] != _ipckg_splt[i]:
                                _flag_installed = False
                                break
                        if _flag_installed:
                            missing_packages.remove(_mpckg)
            elif type(_mpckg) is str:
                if _mpckg == _ipckg["name"]:
                    # print(f">{_mpckg}")
                    missing_packages.remove(_mpckg)

    # Return packages that where not found
    return missing_packages


# packages = [('numpy', '1.19.2'), ('matplotlib', '3.3.2'), ('opencv-python')]
# packages = [('numpy', '1.19'), ('matplotlib', '3.3'), ('opencv-python')]
# packages = [('numpy', '1'), ('matplotlib', '3'), ('opencv-python')]
# packages = [('numpy', '1.19.3'), ('matplotlib', '3.4.2'), ('opencv-python')]
# packages = [('numpy', '2.19'), ('matplotlib', '4'), ('not_existing_package')]
# print(checkPackageAndVersion(packages))


def makeFilter(x_vals: Union[list, np.array] = None,
               y_vals: Union[list, np.array] = None) -> np.array:
    if (x_vals is not None) and (y_vals is not None):
        kernel = np.meshgrid(x_vals, y_vals)
        kernel = np.multiply(kernel[0], kernel[1])
    elif (x_vals is None) and (y_vals is not None):
        kernel = np.meshgrid(np.ones(len(y_vals)), y_vals)[1]
    elif (x_vals is not None) and (y_vals is None):
        kernel = np.meshgrid(x_vals, np.ones(len(x_vals)))[0]
    else:
        kernel = None

    return kernel


def load_image_RGB(path: str):
    from os.path import exists
    from cv2 import imread
    from cv2 import cvtColor
    from cv2 import COLOR_BGR2RGB
    assert exists(path), path + " not found!"
    return cvtColor(imread(path), COLOR_BGR2RGB)


