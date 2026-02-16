#!/bin/bash


if [ -d "venv" ]; then
    echo "Virtual environment found. Activating..."
    source "venv/bin/activate"
    pip install -r requirements.txt
else
    echo "No virtual environment found. Creating one..."
    bash make-venv.sh
fi