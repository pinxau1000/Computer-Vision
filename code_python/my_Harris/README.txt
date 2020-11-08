A SIMPLE MENU IS AVAILABLE ON THE PARENT FOLDER!

The modules are configured to run as a Command Line Interfaces (CLI).
Its also possible to execute individual functions on a IDE like VSCode or
PyCharm.

To run my-harris.py run the command:
>> python -i my-harris.py

To run my-harris-compare.py run the command:
>> python -i my-harris-compare.py

See the parameters that can be configured:
>> python my-harris.py --help
or:
>> python my-harris-compare.py --help

Examples:
>> python -i my-harris.py
>> python -i my-harris.py --bsize 3 --ksize 5 --k 0.04 --threshold 0.01
>> python -i my-harris-compare.py --bsize 3 --ksize 5 --k 0.04 --threshold 0.01

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
