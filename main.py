from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI(title="Plant Disease Classifier API")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
MODEL_PATH = "plant_village_model.keras"
model = None

# PlantVillage disease class names (38 classes)
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

class TFSavedModelWrapper:
    def __init__(self, model_path):
        self.model = tf.saved_model.load(model_path)
        self.serve = self.model.serve
    
    def predict(self, input_data, verbose=0):
        # input_data is numpy array, convert to tensor
        input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
        output = self.serve(input_tensor)
        return output.numpy()

@app.on_event("startup")
async def load_model():
    """Load the model on startup"""
    global model
    try:
        # 1. Try loading from 'model_deployment' directory (SavedModel format)
        # This is the most robust method for deployment
        if os.path.exists("model_deployment"):
            print("Found 'model_deployment' directory. Loading SavedModel...")
            try:
                model = TFSavedModelWrapper("model_deployment")
                print("Successfully loaded SavedModel from 'model_deployment'.")
                return
            except Exception as sm_error:
                print(f"Failed to load SavedModel: {sm_error}")
                print("Falling back to .keras file...")

        # 2. Fallback to loading .keras file
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
        print(f"Loading model from {MODEL_PATH}...")
        
        try:
            # Try standard load first
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print(f"Model loaded successfully using standard load_model.")
        except Exception as load_error:
            print(f"Standard load failed: {load_error}")
            print("Attempting manual reconstruction as Sequential model...")
            
            # Reconstruct as Sequential to match the saved model structure
            # This fixes issues where Functional reconstruction mismatches Sequential weights
            try:
                base_model = tf.keras.applications.Xception(
                    include_top=False,
                    weights=None,
                    input_shape=(224, 224, 3)
                )
                
                model = tf.keras.Sequential([
                    base_model,
                    tf.keras.layers.GlobalAveragePooling2D(name='global_average_pooling2d_5'),
                    tf.keras.layers.Dense(256, activation='relu', name='dense_5'),
                    tf.keras.layers.Dropout(0.5, name='dropout_5'),
                    tf.keras.layers.Dense(38, activation='softmax', name='dense_6')
                ])
                
                # Build to initialize
                model.build((None, 224, 224, 3))
                
                # Load weights
                model.load_weights(MODEL_PATH)
                print(f"Model weights loaded successfully from {MODEL_PATH} into reconstructed Sequential model")
            except Exception as build_error:
                print(f"Manual reconstruction failed: {build_error}")
                raise load_error

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the image to match the model's expected input format.
    The model expects 224x224 RGB images with pixel values normalized to [0, 1].
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 224x224 (the size used during training)
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Plant Disease Classifier API",
        "endpoints": {
            "/predict": "POST - Upload an image to get disease prediction",
            "/health": "GET - Check API health",
            "/classes": "GET - Get list of all disease classes"
        }
    }


@app.get("/health")
async def health_check():
    """Check if the API and model are loaded properly"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.get("/classes")
async def get_classes():
    """Get the list of all plant disease classes"""
    return {"classes": CLASS_NAMES, "total_classes": len(CLASS_NAMES)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict plant disease from an uploaded image.
    
    Args:
        file: Image file (JPG, PNG, etc.)
    
    Returns:
        JSON with prediction results including top predictions and confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get prediction probabilities
        probabilities = predictions[0]
        
        # Get top 5 predictions
        top_indices = np.argsort(probabilities)[-5:][::-1]
        top_predictions = [
            {
                "class": CLASS_NAMES[idx],
                "confidence": float(probabilities[idx]),
                "confidence_percentage": f"{float(probabilities[idx]) * 100:.2f}%"
            }
            for idx in top_indices
        ]
        
        # Get the top prediction
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        
        return JSONResponse(content={
            "success": True,
            "prediction": {
                "class": predicted_class,
                "confidence": confidence,
                "confidence_percentage": f"{confidence * 100:.2f}%"
            },
            "top_5_predictions": top_predictions
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


# Mount static files (for serving the frontend)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    print("Starting Plant Disease Classifier API...")
    print(f"Model path: {MODEL_PATH}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
