
# Opencv-healthcare
Algorithms to digitize Healthcare use-cases (Chair Sit-Stand, Gait Speed Walk Test, Timed Up and Go) for frailty. The eventual deployment of these algorithms will be through a cross-platform mobile application.
## Tech Stack:

CV Functionality:
- Algorithms written in Python and C++17
- Opencv
- MediaPipe for Pose estimation
- Cython


Build tools:
- Cmake

Virtual Environment:
- Running Python3.11



## File Directory

**external/:** Contains external dependencies and submodules.
- pybind11/: Submodule for pybind11, a lightweight header-only library that exposes C++ types in Python and vice versa.

**python_arm_curler/:** Contains specific algorithms or modules related to the project, possibly for arm curl exercises.

**sit_stand_algorithm/:** Contains specific algorithms or modules related to sit-stand exercises.

**src/:** Contains C++ that is used to call the CV algorithms written in Python.

**CMakeLists.txt:** The CMake configuration file used for generating build files and managing the build process.

**requirements.txt:** List of python dependencies used and versions

## Run Locally
Prerequisites:
- Python 3.11 installed

#### Clone repository
```bash
    git clone https://github.com/brennanleez-coder/opencv-healthcare.git

    cd opencv-healthcare
```
#### Setup virtual environment
Name of virtual environment should be myenv
```bash
    python3 -m venv myenv
```

#### Activate virtual environment
```bash
    source myenv/bin/activate
```

#### Install python requirements
This command downloads all required python3 libraries including opencv and mediapipe
```bash
    pip install -r requirements.txt
```

#### Install git submodules (Pybind11)
Pybind11 is used to call python files from a C++ environment
```
    git submodule update --init --recursive
```
Ensure external/pybind11 folder is created with contents


#### Generate executable with Cmake
From the root directory
```bash
    mkdir -p build && cd build

    cmake ..

    make
```

#### Download Test videos
[Test Videos](https://drive.google.com/drive/folders/1508TJTl65lPUibJI231O73kkHrnH0uiE?usp=sharing)
Place them into the test/ directory

#### Run CV Algorithms from C++ main executable
Run the main executable from build directory
```bash
    ./main
``` 
### Development Process

The development process for this project involves several key steps, starting with the development and testing of computer vision algorithms in Jupyter Notebook and transitioning to an integrated C++ and Python environment. Below are the detailed steps to follow:

#### 1. Develop and Test Algorithms in Jupyter Notebook

- Begin by developing and testing the computer vision (CV) algorithms in Jupyter Notebook. This allows for an interactive environment where you can quickly iterate and visualize the results.

#### 2. Convert Notebook to Python Script

- Once the algorithm works as expected in the notebook, convert the Jupyter Notebook to a Python script. This can be done using the following command:
```bash
    jupyter nbconvert --to script example_notebook.ipynb
```

#### 3. Refactor Single python script
- Refactor the generated Python script to follow Object-Oriented Programming (OOP) principles. 
This refactoring helps in organizing the code better and makes it easier to integrate with other components of the project.
Structuring the code in an OOP manner will facilitate navigation and usage of Python directories from the C++ side programmatically.



#### Updated workflow:
Converting code to cython
- Cython is a superset of Python that allows for the writing of C extensions for Python. It is used to speed up Python code by converting it to C code.
- To convert Python code to Cython, create a .pyx file and write the code in Cython syntax. Then, create a setup.py file to compile the Cython code into a shared library.
- To compile the Cython code and move the shared library to the Python path, run the following command:
```bash
    ./Cython_algorithms/compile_and_move.sh
```
Check that there are no errors, warnings are fine.
A build_output.log file will be generated in the Cython_algorithms directory. This file contains the output of the compilation process and can be used to debug any issues that arise during the compilation process.

Errors that are fine:
    PyErr_SetString(PyExc_ZeroDivisionError, "float division");
Would appreciate any help to fix this error. On second run of compile_and_move.sh, the error disappears.


In the setup file:
1. include the full path to the Python.h file in the include_dirs list.
2. include the full path to the .pyx file in the ext_modules list.
This is necessary for the Cython code to compile successfully.

Extra links:
https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/human_pose_estimation_demo/cpp/README.md

Client Server approach:
server/ contains fastapi app

use the following command to run the server locally:
```bash
    fastapi dev server/app/main
```