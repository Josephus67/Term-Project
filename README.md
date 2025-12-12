# Plant Disease Classifier

A full-stack web application for identifying plant diseases using deep learning. Upload an image of a plant leaf, and the AI model will analyze it to detect diseases or confirm if the plant is healthy.

## 🌿 Features

- **38 Disease Classes**: Identifies diseases across multiple plant species including Apple, Tomato, Potato, Corn, Grape, and more
- **Deep Learning Model**: Uses Xception architecture with transfer learning
- **Real-time Predictions**: Fast inference with confidence scores
- **User-Friendly Interface**: Beautiful, responsive web interface
- **Top 5 Predictions**: Shows the most likely conditions with confidence percentages

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- The trained model file: `plant_village_model.keras`

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd Cornelius_Porject
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the FastAPI backend server:**
   ```bash
   python main.py
   ```
   
   The server will start on `http://localhost:8000`

2. **Open the frontend:**
   
   Open your web browser and navigate to:
   ```
   http://localhost:8000/static/index.html
   ```

3. **Upload an image:**
   - Click the upload area or drag and drop a plant leaf image
   - Click "Analyze Plant" to get predictions
   - View the diagnosis and confidence scores

## 📁 Project Structure

```
Cornelius_Porject/
├── main.py                          # FastAPI backend server
├── plant_village_model.keras        # Trained Keras model
├── requirements.txt                 # Python dependencies
├── plant-village-corneliusf2-o.ipynb  # Training notebook
├── static/
│   ├── index.html                  # Frontend HTML
│   ├── style.css                   # Styling
│   └── script.js                   # Frontend JavaScript
└── README.md                       # This file
```

## 🔌 API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /classes` - List all disease classes
- `POST /predict` - Upload image for prediction
- `GET /static/index.html` - Web interface

## 🌱 Supported Plants

- 🍎 Apple
- 🫐 Blueberry
- 🍒 Cherry
- 🌽 Corn (Maize)
- 🍇 Grape
- 🍊 Orange
- 🍑 Peach
- 🫑 Pepper (Bell)
- 🥔 Potato
- 🍓 Strawberry
- 🍅 Tomato
- And more...

## 🧠 Model Details

- **Architecture**: Xception (Transfer Learning)
- **Input Size**: 224x224 RGB images
- **Dataset**: PlantVillage Dataset
- **Classes**: 38 (various diseases + healthy conditions)
- **Preprocessing**: Images are resized to 224x224 and normalized to [0, 1]

## 🛠️ Technology Stack

**Backend:**
- FastAPI
- TensorFlow/Keras
- Uvicorn
- PIL (Pillow)
- NumPy

**Frontend:**
- HTML5
- CSS3
- Vanilla JavaScript
- Responsive Design

## 📝 Usage Example

1. Take or download a photo of a plant leaf
2. Upload it through the web interface
3. The model will analyze the image
4. View the predicted disease with confidence score
5. Check the top 5 possible conditions

## ⚠️ Notes

- The model expects clear images of plant leaves
- For best results, use well-lit photos with the leaf clearly visible
- The model was trained on the PlantVillage dataset
- Confidence scores indicate the model's certainty about the prediction

## 🐛 Troubleshooting

**"Model not loaded" error:**
- Ensure `plant_village_model.keras` is in the project directory

**"Connection refused" error:**
- Make sure the FastAPI server is running on port 8000
- Check if another application is using port 8000

**Low prediction confidence:**
- Try using a clearer, better-lit image
- Ensure the plant leaf is clearly visible
- Make sure the plant species is in the supported list

## 📄 License

This project uses the PlantVillage Dataset and is intended for educational purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 👏 Acknowledgments

- PlantVillage Dataset
- Xception Model (Transfer Learning)
- TensorFlow/Keras Community
