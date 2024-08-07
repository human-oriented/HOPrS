#!/bin/bash

# Install necessary libraries
apt-get update
apt-get install -y libgl1-mesa-glx
apt-get install -y libmagic1

# Run the application
gunicorn --bind=0.0.0.0:8000 app:app
