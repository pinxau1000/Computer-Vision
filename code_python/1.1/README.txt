A SIMPLE MENU IS AVAILABLE ON THE PARENT FOLDER!

The modules are configured to run as a Command Line Interfaces (CLI).
You can start the noise-remove.py module that wraps all the individual functions
or run each function individually. Its also possible to execute individual
functions on a IDE like VSCode or PyCharm.

To run noise-remove.py run the command. This will show all the functions
available by this module.
>> python noise-remove.py

To use the mean-filter you should specify that function:
>> python noise-remove.py mean-filter

or even to get a full list of parameters that you can configure:
>> python noise-remove.py mean-filter --help

Examples:
>> python noise-remove.py original-pictures
>> python noise-remove.py original-pictures --orig ..\..\data\Harris.jpg
>> python noise-remove.py gaussian-filter-sigma --sigma_x [0.25,0.5] --sigma_y [0.1,1] --crop_corner [10,10] --crop_size 16
>> python noise-remove.py bilateral-filter --diameters [5,15] --sigma_c 200

IMPORTANT:
• ALWAYS READ THE DOCUMENTATION (--help)!
• Every function was tested in Python 3.8.6. Please refer to
  https://github.com/pinxau1000/Computer-Vision for suggestions, bug
  and error reports!
