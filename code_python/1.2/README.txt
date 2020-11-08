A SIMPLE MENU IS AVAILABLE ON THE PARENT FOLDER!

The modules are configured to run as a Command Line Interfaces (CLI).
You can start the edge-extract.py module that wraps all the individual functions
or run each function individually.
Its also possible to execute individual functions on a IDE like VSCode or
PyCharm.

To run edge-extract.py run the command:
>> python edge-extract.py

This command without parameters will show an help message equivalent to:
>> python edge-extract.py --help

To use the sobel-filter you should specify that action:
>> python -i edge-extract.py sobel-filter

or even to get a full list of parameters that you can configure:
>> python edge-extract.py sobel-filter --help

Examples:
>> python -i edge-extract.py sobel-filter
>> python -i edge-extract.py sobel-filter --filter_params [6,200,20]
>> python -i edge-extract.py sobel-filter-ddepth --ksize 3 --threshold 0.25
>> python -i edge-extract.py canny-filter --filter_params [15,15]
>> python -i edge-extract.py canny-filter-animate

IMPORTANT:
• Python must be executed in INTERACTIVE MODE (-i flag) so the figures windows
don't close when the program finishes execution.
• When running on INTERACTIVE MODE an CLICK EXCEPTION is HANDLED when the
execution of the script is finished. IGNORE THIS EXCEPTION, press CTRL+Z to EXIT
 INTERACTIVE MODE. When exiting interactive mode all opened figure windows will
 be closed.
• ALWAYS READ THE DOCUMENTATION (--help)!
• Every function was tested in Python 3.8.6. Please refer to https://github.com/pinxau1000
for sugestions, bug and error reports!
