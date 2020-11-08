A SIMPLE MENU IS AVAILABLE ON THE PARENT FOLDER!

The modules are configured to run as a Command Line Interfaces (CLI).
You can start the noise-remove.py module that wraps all the individual functions
or run each function individually.
Its also possible to execute individual functions on a IDE like VSCode or
PyCharm.

To run noise-remove.py run the command:
>> python noise-remove.py

This command without parameters will show an help message equivalent to:
>> python noise-remove.py --help

To use the mean-filter you should specify that action:
>> python -i noise-remove.py mean-filter

or even to get a full list of parameters that you can configure:
>> python noise-remove.py mean-filter --help

Examples:
>> python -i noise-remove.py original-pictures
>> python -i noise-remove.py original-pictures --orig ..\..\data\Harris.jpg
>> python -i noise-remove.py gaussian-filter-sigma --sigma_x [0.25,0.5] --sigma_y [0.1,1] --crop_corner [10,10] --crop_size 16
>> python -i noise-remove.py bilateral-filter --diameters [5,15] --sigma_c 200

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
