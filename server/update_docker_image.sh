#!/bin/bash


git pull origin main

docker build --no-cache -t frailty-vision-be .

docker tag frailty-vision-be brennanlee/frailty-vision-be:latest

docker push brennanlee/frailty-vision-be:latest

echo "Docker image updated and pushed to Docker Hub successfully."
