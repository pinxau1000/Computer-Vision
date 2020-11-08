A SIMPLE MENU IS AVAILABLE ON THE PARENT FOLDER!

The modules are configured to run as a Command Line Interfaces (CLI).
You can start the corner-extract.py module that wraps all the individual functions
or run each function individually.
Its also possible to execute individual functions on a IDE like VSCode or
PyCharm.

To run edge-extract.py run the command:
>> python corner-extract.py

This command without parameters will show an help message equivalent to:
>> python corner-extract.py --help

To use the harris-detector-bsize you should specify that action:
>> python -i corner-extract.py harris-detector

or even to get a full list of parameters that you can configure:
>> python corner-extract.py harris-detector --help

Examples:
>> python -i corner-extract.py harris-detector
>> python -i corner-extract.py harris-detector-bsize
>> python -i corner-extract.py harris-detector-bsize --bsizes [1,2]
>> python -i corner-extract.py harris-detector-ksize --ksizes [3,5,15]
>> python -i corner-extract.py harris-detector-ksize --filter_params [31,31]
>> python -i corner-extract.py harris-detector-k
>> python -i corner-extract.py harris-detector-animate --bsizes [3,5,7,9,11] --ksizes [5,7,9,15,21] --ks [0,0.01,0.02,0.04,0.06,0.08]

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
