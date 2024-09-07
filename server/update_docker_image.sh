#!/bin/bash


git pull origin main

docker build --no-cache -t empower-vision-backend .

docker tag empower-vision-backend brennanlee/empower-vision-backend:latest

docker push brennanlee/empower-vision-backend:latest

echo "Docker image updated and pushed to Docker Hub successfully."
