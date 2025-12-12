// API endpoint (change this if your backend is hosted elsewhere)
const API_URL = 'http://localhost:8000';

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const resultsSection = document.getElementById('resultsSection');
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');

let selectedFile = null;

// Upload area click handler
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File input change handler
fileInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
});

// Drag and drop handlers
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFileSelect(e.dataTransfer.files[0]);
});

// Handle file selection
function handleFileSelect(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert('Please select a valid image file');
        return;
    }

    selectedFile = file;
    const reader = new FileReader();

    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewSection.style.display = 'block';
        predictBtn.disabled = false;
        resultsSection.style.display = 'none';
    };

    reader.readAsDataURL(file);
}

// Clear button handler
clearBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput.value = '';
    uploadArea.style.display = 'block';
    previewSection.style.display = 'none';
    predictBtn.disabled = true;
    resultsSection.style.display = 'none';
});

// Predict button handler
predictBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Show loading state
    predictBtn.disabled = true;
    btnText.textContent = 'Analyzing...';
    btnLoader.style.display = 'inline-block';

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            throw new Error('Prediction failed');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing image. Please make sure the backend server is running on port 8000.');
    } finally {
        // Reset button state
        predictBtn.disabled = false;
        btnText.textContent = 'Analyze Plant';
        btnLoader.style.display = 'none';
    }
});

// Display results
function displayResults(data) {
    const mainPrediction = document.getElementById('mainPrediction');
    const mainConfidence = document.getElementById('mainConfidence');
    const topPredictions = document.getElementById('topPredictions');

    // Format the class name for better display
    const formatClassName = (name) => {
        return name.replace(/_/g, ' ').replace(/([a-z])([A-Z])/g, '$1 $2');
    };

    // Main prediction
    mainPrediction.textContent = formatClassName(data.prediction.class);
    mainConfidence.textContent = data.prediction.confidence_percentage;

    // Top 5 predictions
    topPredictions.innerHTML = '';
    data.top_5_predictions.forEach((pred, index) => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        item.innerHTML = `
            <span class="prediction-name">${index + 1}. ${formatClassName(pred.class)}</span>
            <span class="prediction-confidence">${pred.confidence_percentage}</span>
        `;
        topPredictions.appendChild(item);
    });

    // Show results section
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Check if API is available on page load
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            console.log('API is healthy and ready');
        }
    } catch (error) {
        console.warn('Backend API not reachable. Make sure to start the FastAPI server.');
    }
}

// Check API health on page load
checkAPIHealth();
