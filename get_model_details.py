import tensorflow as tf
import json

try:
    model = tf.keras.models.load_model('plant_village_model.keras', compile=False)
    print("Model loaded.")
    
    # Get config
    config = model.get_config()
    
    # Print relevant details for reconstruction
    print("\n=== Model Architecture Details ===")
    for layer in model.layers:
        print(f"Layer: {layer.name}, Type: {type(layer).__name__}")
        if hasattr(layer, 'get_config'):
            layer_config = layer.get_config()
            if 'units' in layer_config:
                print(f"  Units: {layer_config['units']}")
            if 'rate' in layer_config:
                print(f"  Rate: {layer_config['rate']}")
            if 'activation' in layer_config:
                print(f"  Activation: {layer_config['activation']}")
                
    print("\n=== Xception Details ===")
    # Check if the first layer is Xception
    first_layer = model.layers[0]
    if 'xception' in first_layer.name.lower():
        print("First layer is Xception base.")
        # We need to know if it was instantiated with specific args
        
except Exception as e:
    print(f"Error: {e}")
