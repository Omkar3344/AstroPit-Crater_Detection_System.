from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.exceptions import HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os
from ultralytics import YOLO

# Load the YOLO model
model_path = os.path.abspath("best.pt")  # Get the absolute path
model = YOLO(model_path)  # Load the model

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

@app.get("/")
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect/")
async def detect_objects(file: UploadFile):
    # Get the file size (in bytes)
    file.file.seek(0, 2)
    file_size = file.file.tell()
    
    # Move the cursor back to the beginning
    await file.seek(0)
    
    if file_size > 2 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Should be less than 2MB")
    
    # Check the content type (MIME type)
    content_type = file.content_type
    if content_type not in ["image/jpeg", "image/png", "image/gif"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Should be jpg or png")
    
    # Read the image file
    image_bytes = await file.read()
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Perform detection
    detections = model.predict(image, save=True, conf=0.1)
    output_path = os.path.join(detections[0].save_dir, os.listdir(detections[0].save_dir)[-1])
    
    return FileResponse(output_path)
