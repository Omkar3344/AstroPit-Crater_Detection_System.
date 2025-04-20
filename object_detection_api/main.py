from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Response, Header
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import numpy as np
import shutil
import uuid
from pathlib import Path
import sys
import traceback
import pickle
from ultralytics import YOLO
import logging
from typing import Optional
import re
import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configure paths and directories
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_FOLDER = STATIC_DIR / "uploads"
RESULT_FOLDER = STATIC_DIR / "results"

# Create directories if they don't exist
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)
RESULT_FOLDER.mkdir(exist_ok=True, parents=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Define the FallbackClassifier class to avoid pickle error
class FallbackClassifier:
    """Simple classifier that can be used when loading pickle files with this class."""
    def __init__(self, model=None):
        self.model = model
        
    def predict(self, features):
        """Fallback prediction method."""
        # Default to predicting all inputs as planetary surfaces (class 1)
        return np.ones(len(features), dtype=int)
        
    def predict_proba(self, features):
        """Fallback probability prediction method."""
        # Default to high probability for planetary surfaces
        return np.array([[0.1, 0.9] for _ in range(len(features))])

class CraterDetector:
    def __init__(self):
        # Model paths (adjust as needed)
        self.model_path = str(BASE_DIR / "best.pt")
        logger.info(f"Loading YOLOv8 model from: {self.model_path}")
        
        # Check if model file exists
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found at: {self.model_path}")
            available_files = os.listdir(str(BASE_DIR))
            logger.info(f"Available files in directory: {available_files}")
            self.model = None
        else:
            # Load YOLOv8 model with lower confidence
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                logger.info("YOLOv8 model loaded successfully!")
            except Exception as e:
                logger.error(f"Error loading YOLOv8 model: {e}")
                logger.error(traceback.format_exc())
                self.model = None
            
        # Load planetary surface classifier
        try:
            with open(BASE_DIR / "planetary_classifier.pkl", 'rb') as f:
                self.classifier = pickle.load(f)
            print("Planetary surface classifier loaded successfully!")
        except AttributeError as e:
            if "FallbackClassifier" in str(e):
                print("Using default planetary classifier due to missing class definition")
                self.classifier = FallbackClassifier()
            else:
                print(f"Error loading classifier: {e}")
                self.classifier = None
        except Exception as e:
            print(f"Error loading classifier: {e}")
            self.classifier = None
    
    def extract_features(self, img_path):
        """Extract features for planetary surface classification"""
        try:
            # If the image is a numpy array, save it temporarily
            if isinstance(img_path, np.ndarray):
                temp_path = 'temp_image.jpg'
                cv2.imwrite(temp_path, img_path)
                img_path = temp_path
                
            # Read and process image
            img = cv2.imread(img_path)
            if img is None:
                return None
                
            # Resize to a standard size
            img = cv2.resize(img, (128, 128))
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Extract simple features
            features = []
            
            # Color histograms (RGB channels)
            for i in range(3):
                hist = cv2.calcHist([img], [i], None, [32], [0, 256])
                features.extend(hist.flatten())
            
            # Texture features
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            features.append(np.mean(laplacian))
            features.append(np.std(laplacian))
            
            # Edge density
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges) / (128 * 128)
            features.append(edge_density)
            
            # Average brightness
            features.append(np.mean(gray))
            
            # Clean up temp file if needed
            if img_path == 'temp_image.jpg' and os.path.exists(temp_path):
                os.remove(temp_path)
                
            return np.array(features)
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def is_planetary_surface(self, image_path):
        """Check if image is a planetary surface"""
        if self.classifier is None:
            return True  # Default to True if no classifier available
            
        features = self.extract_features(image_path)
        if features is None:
            return False
            
        try:
            prediction = self.classifier.predict_proba([features])[0]
            return prediction[1] > 0.5  # Threshold for planetary surface
        except:
            # Fallback for simpler classifier implementations
            prediction = self.classifier.predict([features])[0]
            return prediction == 1
    
    def detect_craters(self, image_path, conf=0.1):
        """Detect craters in an image with planetary surface validation"""
        if self.model is None:
            logger.error("Model not loaded, cannot perform detection")
            img = np.zeros((400, 600, 3), dtype=np.uint8)
            cv2.putText(img, "Model not loaded", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return img, "Model not loaded"
        
        # Load the image
        try:
            if isinstance(image_path, str):
                img = cv2.imread(image_path)
                if img is None:
                    logger.error(f"Could not read image at: {image_path}")
                    return np.zeros((400, 600, 3), dtype=np.uint8), "Could not read image"
            else:
                img = image_path.copy()  # Make a copy to avoid modifying the original
            
            # Report image details for debugging
            logger.info(f"Image shape: {img.shape}, dtype: {img.dtype}")
            
            # First check if the image is a planetary surface
            is_planetary = self.is_planetary_surface(image_path)
            logger.info(f"Is planetary surface: {is_planetary}")
            
            if not is_planetary:
                # Return the original image with a warning message
                warning_img = img.copy()
                cv2.putText(
                    warning_img,
                    "NOT A PLANETARY SURFACE - No crater detection performed",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                return warning_img, "Not a planetary surface"
            
            # Only run detection if it's a planetary surface
            logger.info(f"Running detection with confidence threshold: {conf}")
            results = self.model.predict(img, conf=conf, verbose=True)
            logger.info(f"Detection completed, results: {results}")
            
            # Rest of your detection code remains the same
            annotated_img = results[0].plot()
            
            # Get detection results
            detections = []
            if len(results[0].boxes) > 0:
                logger.info(f"Found {len(results[0].boxes)} potential craters")
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    
                    logger.info(f"Detection: box={[x1, y1, x2, y2]}, confidence={confidence:.4f}, class={class_id}")
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(confidence),
                        'class_id': class_id
                    })
            else:
                logger.warning("No craters detected in the image")
                # Add text to indicate no detections
                cv2.putText(
                    annotated_img, 
                    "No craters detected - try adjusting settings", 
                    (30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2
                )
                    
            return annotated_img, detections
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return original image with error text
            if isinstance(image_path, str) and os.path.exists(image_path):
                img = cv2.imread(image_path)
            elif isinstance(image_path, np.ndarray):
                img = image_path.copy()
            else:
                img = np.zeros((400, 600, 3), dtype=np.uint8)
                
            cv2.putText(img, f"Error: {str(e)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return img, f"Error during detection: {str(e)}"

# Initialize detector
detector = CraterDetector()

@app.get("/")
async def read_root():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/favicon.ico")
async def get_favicon():
    """Serve the favicon"""
    favicon_path = STATIC_DIR / "favicon.ico"
    if not favicon_path.exists():
        # Return a 204 No Content if the favicon doesn't exist
        return Response(status_code=204)
    return FileResponse(favicon_path)

@app.post("/detect/")
async def detect_craters(file: UploadFile = File(...)):
    """Detect craters in an uploaded image"""
    # Validate file
    if not file.filename:
        return JSONResponse(status_code=400, content={"detail": "No file provided"})
    
    # Create a detailed log of what we're processing
    logger.info(f"Processing file: {file.filename} (content-type: {file.content_type})")
    
    try:
        # Check file extension
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ""
        logger.info(f"File extension: {file_ext}")
        
        # Validate file type
        if file_ext not in ALLOWED_EXTENSIONS:
            return JSONResponse(
                status_code=400, 
                content={"detail": f"Invalid file type. Supported formats: {', '.join(ALLOWED_EXTENSIONS)} (images only)"}
            )
        
        # Process image file
        if file_ext in ['jpg', 'jpeg', 'png', 'gif']:
            # Create temp path and save file
            unique_id = str(uuid.uuid4())
            temp_path = UPLOAD_FOLDER / f"temp_{unique_id}.{file_ext}"
            
            # Save the uploaded file
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            try:
                # Save original image
                original_filename = f"original_{unique_id}.{file_ext}"
                original_path = RESULT_FOLDER / original_filename
                shutil.copy(str(temp_path), str(original_path))
                
                # Use the crater detector to process the image
                logger.info("Processing image with crater detector...")
                annotated_img, detections = detector.detect_craters(str(temp_path), conf=0.05)  # Lower threshold to 0.05

                # Save the result image filename (defined before we might need it)
                result_filename = f"result_{unique_id}.{file_ext}"
                result_path = RESULT_FOLDER / result_filename

                # Then check if detections is a string message (error case)
                if isinstance(detections, str) and "Not a planetary surface" in detections:
                    # Now result_filename is defined
                    return JSONResponse(content={
                        "success": False,
                        "message": "Not a planetary surface - invalid image",
                        "result_type": "image",
                        "original_path": f"/static/results/{original_filename}",
                        "result_path": f"/static/results/{result_filename}",
                        "original_download": f"/download-image/original/{original_filename}",
                        "result_download": f"/download-image/result/{result_filename}"
                    })
                
                # Save the result image
                cv2.imwrite(str(result_path), annotated_img)
                
                # If detections is a list, extract more detailed information
                if isinstance(detections, list):
                    # Add image dimensions to the response
                    image_height, image_width = annotated_img.shape[:2]
                    
                    # Get the filename
                    image_name = os.path.basename(original_filename)
                    
                    # Return enhanced response with more details
                    return JSONResponse(content={
                        "success": True,
                        "message": f"Found {len(detections)} craters",
                        "result_type": "image",
                        "image_name": image_name,
                        "image_dimensions": {
                            "width": image_width,
                            "height": image_height
                        },
                        "original_path": f"/static/results/{original_filename}",
                        "result_path": f"/static/results/{result_filename}",
                        "original_download": f"/download-image/original/{original_filename}",
                        "result_download": f"/download-image/result/{result_filename}",
                        "detections": detections,
                        "detection_time": datetime.datetime.now().isoformat()
                    })
            
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                logger.error(traceback.format_exc())
                return JSONResponse(
                    status_code=500,
                    content={"detail": f"Error processing image: {str(e)}"}
                )
            
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        else:
            return JSONResponse(
                status_code=400,
                content={"detail": "Unsupported file type"}
            )
    
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Server error: {str(e)}"}
        )

@app.get("/download-image/{image_type}/{filename}")
async def download_image(image_type: str, filename: str):
    """Download image with the proper content type and download headers"""
    if image_type not in ["original", "result"]:
        return JSONResponse(status_code=400, content={"detail": "Invalid image type"})
    
    image_path = RESULT_FOLDER / filename
    
    if not image_path.exists():
        return JSONResponse(status_code=404, content={"detail": "Image not found"})
    
    # Determine content type based on extension
    ext = filename.split('.')[-1].lower()
    content_type = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif'
    }.get(ext, 'image/jpeg')
    
    return FileResponse(
        path=str(image_path),
        media_type=content_type,
        filename=f"crater_detection_{image_type}.{ext}"
    )

# Add a simple test endpoint to check if server is working
@app.get("/test")
async def test_server():
    """Simple endpoint to test if server is working"""
    return {"status": "ok", "message": "Server is running"}

# Add this function to check available models
@app.get("/list-models")
async def list_models():
    """List all available model files in the project directory"""
    model_files = []
    
    # Check main directory
    main_dir = Path(__file__).parent
    for file in main_dir.glob("*.pt"):
        model_files.append(str(file.relative_to(main_dir)))
    
    # Check models directory if it exists
    models_dir = main_dir / "models"
    if models_dir.exists():
        for file in models_dir.glob("*.pt"):
            model_files.append(f"models/{file.relative_to(models_dir)}")
    
    return {"available_models": model_files}

# Add this endpoint to test detection with different confidence thresholds
@app.get("/test-detection")
async def test_detection(conf: float = 0.1, model: str = None):
    """Test crater detection with sample images and the specified confidence threshold"""
    
    # Check if we should use a specific model
    if model:
        model_path = BASE_DIR / model
        if not model_path.exists():
            return JSONResponse(
                status_code=400, 
                content={"detail": f"Model file not found: {model}"}
            )
        
        try:
            from ultralytics import YOLO
            test_model = YOLO(str(model_path))
            logger.info(f"Test model loaded from: {model_path}")
        except Exception as e:
            logger.error(f"Error loading test model: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": f"Error loading model: {str(e)}"}
            )
    else:
        # Use the existing detector's model
        test_model = detector.model
        if test_model is None:
            return JSONResponse(
                status_code=500,
                content={"detail": "No model loaded in detector"}
            )
    
    # Find sample images to test
    sample_images = []
    for ext in ['jpg', 'jpeg', 'png']:
        sample_images.extend(list(STATIC_DIR.glob(f"samples/*.{ext}")))
    
    if not sample_images and RESULT_FOLDER.exists():
        # Look in results folder if no samples
        for ext in ['jpg', 'jpeg', 'png']:
            sample_images.extend(list(RESULT_FOLDER.glob(f"original_*.{ext}")))
    
    if not sample_images:
        return JSONResponse(
            status_code=404,
            content={"detail": "No sample images found for testing"}
        )
    
    # Use the first sample image
    test_image = str(sample_images[0])
    logger.info(f"Testing detection with image: {test_image}")
    
    try:
        # Run detection with the specified confidence
        results = test_model.predict(test_image, conf=conf, verbose=True)
        
        # Process results
        if len(results) == 0:
            return {"status": "error", "message": "No results returned from model"}
        
        boxes = results[0].boxes
        detections = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'confidence': float(confidence),
                'class_id': class_id
            })
        
        # Save result image for verification
        result_img = results[0].plot()
        result_path = RESULT_FOLDER / "test_detection_result.jpg"
        cv2.imwrite(str(result_path), result_img)
        
        return {
            "status": "success",
            "detections": detections,
            "detection_count": len(detections),
            "confidence_threshold": conf,
            "result_image": "/static/results/test_detection_result.jpg"
        }
    except Exception as e:
        logger.error(f"Test detection error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Detection failed: {str(e)}"}
        )

# Add this endpoint to debug model issues
@app.get("/debug-model")
async def debug_model():
    """Debug the YOLO model to see if it's properly loaded"""
    if detector.model is None:
        return {"status": "error", "message": "Model not loaded"}
    
    try:
        # Get model info
        model_info = {
            "model_path": detector.model_path,
            "model_exists": os.path.exists(detector.model_path),
            "model_size_bytes": os.path.getsize(detector.model_path) if os.path.exists(detector.model_path) else 0,
            "model_type": str(type(detector.model)),
            "model_classes": detector.model.names if hasattr(detector.model, "names") else {},
            "model_loaded": detector.model is not None
        }
        
        # Try a simple prediction on a colored square
        test_img = np.ones((300, 300, 3), dtype=np.uint8) * 128  # Gray image
        # Draw a circle to see if detection works on basic shapes
        cv2.circle(test_img, (150, 150), 50, (0, 0, 255), -1)
        
        # Save this test image
        test_img_path = RESULT_FOLDER / "test_image.jpg"
        cv2.imwrite(str(test_img_path), test_img)
        
        # Try to run prediction
        results = detector.model.predict(str(test_img_path), conf=0.01, verbose=True)
        
        # Check results
        prediction_info = {
            "result_count": len(results),
            "boxes_count": len(results[0].boxes) if len(results) > 0 else 0,
            "first_result_shape": results[0].boxes.shape if len(results) > 0 and len(results[0].boxes) > 0 else None
        }
        
        return {
            "status": "success", 
            "model_info": model_info,
            "prediction_test": prediction_info,
            "test_image": "/static/results/test_image.jpg"
        }
    except Exception as e:
        logger.error(f"Error in debug-model: {e}")
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...)):
    """Upload a new YOLO model file"""
    if not file.filename.endswith('.pt'):
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid file type. Only .pt model files are supported."}
        )
    
    try:
        # Save the model file
        model_path = BASE_DIR / "custom_model.pt"
        with open(model_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Reload the detector with the new model
        try:
            from ultralytics import YOLO
            new_model = YOLO(str(model_path))
            detector.model = new_model
            detector.model_path = str(model_path)
            
            return {"status": "success", "message": "Model uploaded and loaded successfully"}
        except Exception as e:
            logger.error(f"Error loading uploaded model: {e}")
            return JSONResponse(
                status_code=500,
                content={"detail": f"Model uploaded but could not be loaded: {str(e)}"}
            )
    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error uploading model: {str(e)}"}
        )

# Replace your current formatJSON function with this optimized version
def formatJSON(json):
    if not json:
        return 'No data available'
    
    # Stringify with indentation (but limit the work for large objects)
    try:
        stringSize = len(JSONResponse(json).body)
        if stringSize > 100000:
            # For very large JSON, use minimal formatting to improve performance
            jsonString = JSONResponse(json, indent=1).body.decode()
        else:
            jsonString = JSONResponse(json, indent=2).body.decode()
    except Exception as e:
        return f"Error formatting JSON: {e}"
    
    # Early return for very large strings - syntax highlighting becomes too expensive
    if len(jsonString) > 200000:
        return jsonString

    # Optimized regex-based syntax highlighting
    return re.sub(
        r'("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)',
        lambda match: f'<span class="json-{match.group(0)}">{match.group(0)}</span>',
        jsonString
    )

# Add this at the bottom of main.py
if __name__ == "__main__":
    import uvicorn
    
    print("Models loaded. Starting FastAPI server...")
    
    # Start the uvicorn server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )