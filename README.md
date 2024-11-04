
# Opencv-healthcare
Algorithms to digitize Healthcare tests (Chair Sit-Stand, Gait Speed Walk Test, Timed Up and Go) for frailty

## Tech Stack:
CV Functionality:
- OpenCV
- MediaPipe for Pose estimation

Virtual Environment:
- Running Python3.11

## File Directory

**external/:** Contains external dependencies and submodules.
- pybind11/: Submodule for pybind11, a lightweight header-only library that exposes C++ types in Python and vice versa.

**python_arm_curler/:** Contains specific algorithms or modules related to the project, possibly for arm curl exercises.

**sit_stand_algorithm/:** Contains specific algorithms or modules related to sit-stand exercises.

**src/:** Contains C++ files that is used to call the CV algorithms written in Python. This process is no longer in use.

**CMakeLists.txt:** The CMake configuration file used for generating build files and managing the build process.

**requirements.txt:** List of python dependencies used and versions

## Run Locally

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

#### Install git submodules (Pybind11) (Optional, if calling from C++ environment). This is no longer in use.
Pybind11 is used to call python files from a C++ environment
```
    git submodule update --init --recursive
```
Ensure external/pybind11 folder is created with contents


#### Generate executable with Cmake (Optional, if calling from C++ environment). This is no longer in use.
From the root directory
```bash
    mkdir -p build && cd build

    cmake ..

    make
```

#### Download Test videos
[Test Videos](https://drive.google.com/drive/folders/1508TJTl65lPUibJI231O73kkHrnH0uiE?usp=sharing)
Place them into the test/ directory


### Development Process

#### 1. Develop and Test Algorithms in Jupyter Notebook
-Begin by developing and testing the computer vision (CV) algorithms in Jupyter Notebook. This allows for an interactive environment where you can quickly iterate and visualize the results.

#### 2. Convert to Cython
- Cython is a superset of Python that allows for the writing of C extensions for Python. It is used to speed up Python code by converting it to C code.
- To convert Python code to Cython, create a .pyx file and write the code in Cython syntax. Then, create a setup.py file to compile the Cython code into a shared library.
- To compile the Cython code and move the shared library to the Python path, run the following command:
```bash
    ./Cython_algorithms/compile_and_move.sh
```
Check that there are no errors, warnings are fine.
A build_output.log file will be generated in the Cython_algorithms directory. This file contains the output of the compilation process and can be used to debug any issues that arise during the compilation process.

#### Deploy Algorithms in FastAPI
server/ contains the fastapi app

use the following command to run the server locally:
```bash
    cd server
    fastapi dev
```

Build fastapi app as docker image
```bash
    docker build -t image-name -f Dockerfile .
```
Run the docker image
```bash
    docker run -d --name container-name -p 80:80 image-name
```

#### Compiling for Linux
To compile for linux, I created another docker container just to compile and then copy the .so file locally. Then move it to my fastapi app.
If not done properly, the .so file will not work on the fastapi app that is running inside the docker container.
    
```bash
    docker build cython-app . 
    docker run -d cython-app
```
Check if the file is inside
```bash
    docker exec -it <CONTAINER_ID> /bin/bash
```
Enter the container
```bash
    docker cp CONTAINER_ID:app/File .          
```
Once the sit_stand_overall.cpython-310-x86_64-linux-gnu.so is created, move it to the fastapi app
    
    
