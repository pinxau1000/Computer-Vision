# Computer Vision
### Introduction
Work developed by **José Rosa** and **Ricardo Silva** as 1st Assigment of Computer Vision - MEE - ESTG.

### Images
<table>
  <tr>
    <td>
      <img alt="Image of Mean Filter" src="code_python/1.1/output_MeanFilter.png" width="100%">
    </td>
    <td>
      <img alt="Image of Sobel Filter Magnitude and Orientation" src="code_python/1.2/output_SobelFilter_MagAng.png" width="100%">
    </td>
    <td>
      <img alt="Image of Custom Harris Detector" src="code_python/my_Harris/output_CustomHarrisCornerDetector.png" width="100%">
    </td>
  </tr>
</table>
<table>
  <tr>
   <td>
      <img alt="Image of CLI Help" src="CLI_1.png" width="100%">
    </td>
   <td>
      <img alt="Image of CLI Run" src="CLI_2.png" width="100%">
    </td>
  </tr>
</table>

### Recomendations
###### Run these scripts on a virtual environment [[Official Python Guide Here](https://docs.python.org/3/tutorial/venv.html)]
1. If you don't have Python 3.8 installed we recomend that you install it on your machine! **Don't add it to PATH if you will not use it as you main Python Interpreter.**
   - You can use other version of the Python Interpreter but you may experience compatability issues. If so in the rest of this guide where we refer as Python 3.8 you should refer to your version of Python.
2. Go to the project folder, open CMD/Terminal instance there and run:
   ```sh
   [PATH TO PYTHON 3.8 EXECUTABLE] -m venv venv
   ```
   Example:
   ```sh
   C:\Users\[USER]\AppData\Local\Programs\Python\Python38\python.exe -m venv venv
   ```
   This will create a virtual environment with Python Interpreter 3.8 in the current directory.
3. Run the following command on the console to activate the Virtual Environment:
   ```sh
   venv\Scripts\activate
   ```
4. Run and explore these examples!
   - When running the functions on this repository the required packages should be installed automatically. You can install them manually by running the command ```pip install -r requirements.txt```


### Run
###### Help Commands
Syntax: `python <script_name.py> --help`
```sh
python noise-remove.py --help
```
```sh
python gaussian-filter-sigma.py --help
```
```sh
python edge-extract.py --help
```

###### Example Commands
Syntax: `python <script_name.py> [function] --[parameter] [value]`
```sh
python noise-remove.py gaussian-filter-sigma --sigma_x [0.25,0.5] --sigma_y [0.1,1] --crop_corner [10,10] --crop_size 16
```
```sh
python bilateral-filter.py --diameters [5,15] --sigma_c 200
```
```sh
python edge-extract.py sobel-filter-ddepth --ksize 3 --threshold 0.25
```
```sh
python harris-detector-animate.py --bsizes [3,5,7,9,11] --ksizes [5,7,9,15,21] --ks [0,0.01,0.02,0.04,0.06,0.08]
```

### Known Issues:
- Issue: Figures don't show when running a command.
  - Fix: Run the command with python in Interactive Mode. After execution you can exit Interacrive Mode pressing CTRL+Z. Note that if you exit Interactive Mode all opened figure windows will be closed.
  <img alt="Image of CLI Interactive Mode" src="CLI_4.png" width="33%">
- Issue: Figures windows text is overlapped.
  - This is caused by your resolution being not the same as the authors. You can open the `util.py` script and change the `dpi` parameter.
- Issue: Scripts aren't running. :cry:
  - The scripts were tested using `Python 3.8.6` with the following required packages installed:
    - `click 7.1.2` [download](https://pypi.org/project/click/)
    - `matplotlib 3.3.2` [download](https://pypi.org/project/matplotlib/)
    - `numpy 1.19.2` [download](https://pypi.org/project/numpy/)
    - `opencv-python 4.4.0.44` [download](https://pypi.org/project/opencv-python/)
    - `packaging 20.4` - [download](https://pypi.org/project/packaging/)
- **Read the README.txt files!**

### Structure
 - code_python: All the source code is here
   - 1.1: Source code that implements noise remove algoritms
     - noise-remove.py: Wraps all the scripts.
       Get list of functions: `python noise-remove.py --help`
       Example: 
       ```sh
       python noise-remove.py original-pictures
       ```
   - 1.2: Source code that implemente edge extraction algoritms
     - edge-extract.py: Wraps all the scripts.
       Get list of functions: ```python edge-extract.py --help```
       Example:
       ```sh
       python edge-extract.py sobel-filter
       ```
   - 1.3: Source code that implemente corner extract algoritms
     - corner-extract.py: Wraps all the scripts.
       Get list of functions: ```python corner-extract.py --help```
       Example:
       ```sh
       python corner-extract.py harris-detector
       ```
   - my_Harris:
     - my-harris.py: Custom Harris Implementation.
       Get list of parameters: ```python my-harris.py --help```
       Example:
       ```sh
       python my-harris.py
       ```
     - my-harris-compare.py: Comparison between Custom Harris Implementation and OpenCV Harris.
       Get list of parameters: ```python my-harris-compare.py --help```
       Example:
       ```sh
       python my-harris-compare.py
       ```
   - Easy_Run.py: Prompts a console menu which helps the user to run a desired function.
 - data: Folder with images to use in the functions.
 - papers: References used to build the scripts.

### Contribute
Feel free to contribute, submit sugestions and error reports.

### Licence
MIT LICENSED: It permits users to do anything with a given project as long as they credit the developer and don’t hold him or her liable for the project’s use.
