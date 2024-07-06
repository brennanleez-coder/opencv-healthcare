# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    cython3 \
    libatlas-base-dev  # Ensure necessary libraries for numpy are installed

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Compile the Cython code
RUN python setup_docker.py build_ext --inplace
