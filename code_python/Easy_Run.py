import subprocess
import sys

_PYTHON_INTERPRETER = sys.executable + " -i"
_CURRENT_DIRECTORY = sys.path[0]

_opt = ""

while _opt != '0':
    print("---------------- MAIN MENU ----------------")
    print("1 - Noise Removal")
    print("2 - Edge Extraction")
    print("3 - Corner Detection")
    print("4 - Custom Harris")
    print("0 - Exit")
    print("-------------------------------------------")
    _opt = input("Option:\t")
    _opt = str(_opt)

    if _opt == "1":
        while _opt != "0":
            print("---------------- Noise Remove ----------------")
            print("a - Plot Original Pictures")
            print("b - Use Mean Filter")
            print("c - Visualize Mean Filter Anchor Effect")
            print("d - Use Median Filter")
            print("e - Use Gaussian Filter")
            print("f - Visualize Gaussian Filter Sigma Effect")
            print("g - Use Bilateral Filter")
            print("h - Visualize Bilateral Filter Sigma Effect")
            print("0 - Exit")
            print("----------------------------------------------")
            _opt = input("Option:\t")
            _opt = str(_opt)

            if _opt == "a":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.1/original-pictures.py")
            elif _opt == "b":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.1/mean-filter.py")
            elif _opt == "c":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.1/mean-filter-anchor.py")
            elif _opt == "d":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.1/median-filter.py")
            elif _opt == "e":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.1/gaussian-filter.py")
            elif _opt == "f":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.1/gaussian-filter-sigma.py")
            elif _opt == "g":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.1/bilateral-filter.py")
            elif _opt == "h":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.1/bilateral-filter-sigma.py")
            elif _opt == "0":
                pass
            else:
                print("Invalid Option!")
        _opt = ""
    elif _opt == "2":
        while _opt != "0":
            print("---------------- Edge Extraction ----------------")
            print("a - Visualize Data Type Problems on Edge Detection")
            print("b - Use Sobel Operator")
            print("c - Use Scharr 3x3 Kernel")
            print("d - Use Scharr 3x3 Kernel and Apply Threshold")
            print("e - Use Prewitt Filter")
            print("f - Use Roberts Filter")
            print("g - Use Canny Edge Detector")
            print("h - Generate an Animation of Canny")
            print("i - Use Laplacian Filter")
            print("0 - Exit")
            print("-------------------------------------------------")
            _opt = input("Option:\t")
            _opt = str(_opt)

            if _opt == "a":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.2/sobel-filter-ddepth.py")
            elif _opt == "b":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.2/sobel-filter.py")
            elif _opt == "c":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.2/scharr-filter.py")
            elif _opt == "d":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.2/scharr-filter-threshold.py")
            elif _opt == "e":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.2/prewitt-filter.py")
            elif _opt == "f":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.2/roberts-filter.py")
            elif _opt == "g":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.2/canny-filter.py")
            elif _opt == "h":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.2/canny-filter-animate.py")
            elif _opt == "i":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.2/laplacian-filter.py")
            elif _opt == "0":
                pass
            else:
                print("Invalid Option!")
        _opt = ""
    elif _opt == "3":
        while _opt != "0":
            print("---------------- Corner Detection ----------------")
            print("a - Apply the Harris to an image")
            print("b - Visualize the Effect of Block Size on Harris")
            print("c - Visualize the Effect of Sobel Kernel Aperture on Harris")
            print("d - Visualize the Effect of Harris Free Parameter on Harris")
            print("e - Generate an Animation Sweeping the Harris Parameters")
            print("0 - Exit")
            print("-------------------------------------------------")
            _opt = input("Option:\t")
            _opt = str(_opt)
            if _opt == "a":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.3/harris-detector.py")
            elif _opt == "b":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.3/harris-detector-bsize.py")
            elif _opt == "c":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.3/harris-detector-ksize.py")
            elif _opt == "d":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.3/harris-detector-k.py")
            elif _opt == "e":
                subprocess.run(_PYTHON_INTERPRETER +
                               " 1.3/harris-detector-animate.py")
            elif _opt == "0":
                pass
            else:
                print("Invalid Option!")
        _opt = ""
    elif _opt == "4":
        while _opt != "0":
            print("-------------------- Custom Harris --------------------")
            print("a - Apply the Custom Harris Corner Detector to an image")
            print("b - Compare the OpenCV and Custom Harris results")
            print("0 - Exit")
            print("-------------------------------------------------------")
            _opt = input("Option:\t")
            _opt = str(_opt)
            if _opt == "a":
                subprocess.run(_PYTHON_INTERPRETER +
                               " my_Harris/my-harris.py")
            elif _opt == "b":
                subprocess.run(_PYTHON_INTERPRETER +
                               " my_Harris/my-harris-compare.py")
            elif _opt == "0":
                pass
            else:
                print("Invalid Option!")


        _opt = ""
    elif _opt == "0":
        pass
    else:
        print("Invalid Option!")
