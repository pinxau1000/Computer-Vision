A SIMPLE MENU IS AVAILABLE ON THE PARENT FOLDER!

The modules are configured to run as a Command Line Interfaces (CLI).
You can start the edge-extract.py module that wraps all the individual functions
or run each function individually. Its also possible to execute individual
functions on a IDE like VSCode or PyCharm.

To run edge-extract.py run the command:
>> python edge-extract.py

To use the sobel-filter you should specify that action:
>> python edge-extract.py sobel-filter

or even to get a full list of parameters that you can configure:
>> python edge-extract.py sobel-filter --help

Examples:
>> python edge-extract.py sobel-filter
>> python edge-extract.py sobel-filter --filter_params [6,200,20]
>> python edge-extract.py sobel-filter-ddepth --ksize 3 --threshold 0.25
>> python edge-extract.py canny-filter --filter_params [15,15]
>> python edge-extract.py canny-filter-animate


IMPORTANT:
• ALWAYS READ THE DOCUMENTATION (--help)!
• Every function was tested in Python 3.8.6. Please refer to
  https://github.com/pinxau1000/Computer-Vision for suggestions, bug
  and error reports!

