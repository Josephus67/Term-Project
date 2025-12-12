import tensorflow as tf
import os

print(f"TensorFlow Version: {tf.__version__}")

try:
    model = tf.keras.models.load_model('plant_village_model.keras', compile=False)
    print("Model loaded successfully locally.")
    model.summary()
    
    print("\nAttempting to export to SavedModel format (more robust for Linux deployment)...")
    model.export('model_deployment')
    print("Success! Created 'model_deployment' directory.")
    
except Exception as e:
    print(f"Error: {e}")
