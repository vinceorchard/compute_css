#!/bin/bash

# === 1. Create Virtual Environment ===
echo "Creating virtual environment..."
python3 -m venv venv

# === 2. Activate Virtual Environment ===
# Note: This will only affect the current shell if sourced
echo "Activating virtual environment..."
source "venv/bin/activate"


# === 3. Install dependencies from requirements.txt ===
echo "Installing dependencies from requirements.txt..."
python get-pip-25.0.1
pip install -r requirements.txt
