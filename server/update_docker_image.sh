#!/bin/bash


git pull origin main

docker build -t empower-vision-be .


docker tag empower-vision brennanlee/empower-vision-be:latest

docker push brennanlee/empower-vision-be:latest

echo "Docker image updated and pushed to Docker Hub successfully."
