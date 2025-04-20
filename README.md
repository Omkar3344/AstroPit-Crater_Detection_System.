# Martian & Lunar Crater Detection - Quick Start
### This guide contains all the commands needed to run the Martian & Lunar Crater Detection project.

## Setup Commands
### Clone the repository
git clone https://github.com/yourusername/Martian_Lunar_Crater_Detection.git
cd Martian_Lunar_Crater_Detection

### Create and activate virtual environment
## Windows
### Create virtual environment
python -m venv venv

### Activate virtual environment
venv\Scripts\activate

## macOS/Linux
### Create virtual environment
python3 -m venv venv

### Activate virtual environment
source venv/bin/activate

## Install required packages
pip install -r requirements.txt

## Run the application
### Navigate to the API directory
cd object_detection_api

### Load the models
python main.py

### Start the FastAPI server
uvicorn main:app --reload --log-level debug