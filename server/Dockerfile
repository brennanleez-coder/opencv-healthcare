# Use an official Python runtime as a parent image
FROM tiangolo/uvicorn-gunicorn:python3.10-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    cython3 \
    libatlas-base-dev \
    libgl1-mesa-glx \ 
    libglib2.0-0
    # install the OpenGL library
    #install the GLib library


# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY ../requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the server directory contents into the container at /app
COPY . .

ENV PYTHONPATH=/app

# Make port 80 available to the world outside this container
EXPOSE 8000

# Run app/main.py when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
