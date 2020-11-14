REQUIREMENTS:
- Python >= 3.5
- click>=7.1
- matplotlib>=3.3
- opencv-python>=4.4.0
- packaging>=20.4
- numpy>=1.19,<1.19.4
# There is a error with numpy 1.19.4 on windows. It will be fixed around January 2021.
# https://developercommunity.visualstudio.com/content/problem/1207405/fmod-after-an-update-to-windows-2004-is-causing-a.html

It's recommended to create a virtual environment to run these scripts. You can
 use the Python recommended and default tool. To do so follow these steps:
1. If you don't have Python 3.8 installed on your machine install it. Don't add
it to PATH if you will not use it as you main Python Interpreter.
2. Go to the project folder, open CMD/Terminal there and run:
>> [PATH TO PYTHON 3.8 EXECUTABLE] -m venv venv
Example:
>> C:\Users\[USER]\AppData\Local\Programs\Python\Python38\python.exe -m venv venv
This will create a virtual environment with Python Interpreter 3.8 in the current
 directory.
3. Run the following command on the console to activate the Virtual Environment:
>> venv\Scripts\activate
4. Run and explore these examples
   - When running the functions on this repository the required packages should
   be installed automatically. You can install them manually by running the
   command: >> pip install -r requirements.txt

Easy_Run.py is a simple menu that automatically runs commands in order to execute
the functions with the default parameters.

Easy_Run.py is a simple menu that automatically runs commands in order to execute
the functions with the default parameters.

Start Easy_Run.py:
>> python Easy_Run.py

IMPORTANT:
• DETAILED DOCUMENTATION IS AVAILABLE AT SUB-FOLDERS 1.1, 1.2, 1.3 and
  my_Harris!
• ALWAYS READ THE DOCUMENTATION (--help)!
• Every function was tested in Python 3.8.6. Please refer to
  https://github.com/pinxau1000/Computer-Vision for suggestions, bug
  and error reports!
