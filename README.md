# speech_emotion_recognition1

A Flask-based web application that uses machine learning to recognize emotions from audio files. The system analyzes audio features and predicts emotional states with confidence scores and interactive visualizations.

## 🌟 Features

- **Multi-format Audio Support**: WAV, MP3, FLAC, M4A, OGG
- **Real-time Emotion Detection**: Instant analysis and results
- **Interactive Visualizations**: Bar charts and pie charts showing emotion probabilities
- **Secure File Handling**: Automatic cleanup and secure file processing
- **Responsive Web Interface**: Modern, user-friendly design
- **Confidence Scoring**: Probability scores for all detected emotions

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │ ──▶│  Flask Backend   │ ──▶│  ML Pipeline    │
│   (HTML/JS)     │    │  (File Handling) │    │  (Prediction)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                 │                        │
                                 ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  File Validation │    │  Visualization  │
                       │  & Security      │    │  Generation     │
                       └──────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

### System Requirements
- Python 3.7+
- 4GB+ RAM recommended
- Audio processing libraries support

### Python Dependencies
```bash
flask>=2.0.0
numpy>=1.19.0
scikit-learn>=1.0.0
joblib>=1.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
librosa>=0.8.0  # For audio processing
werkzeug>=2.0.0
```

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/vihabhat/speech_emotion_recognition1.git
cd speech_emotion_recognition1
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Directory Structure
```
emotion-recognition-system/
├── app.py                 # Main Flask application
├── feature_extractor.py   # Audio feature extraction module
├── model_trainer.py       # Model training script
├── requirements.txt       # Python dependencies
├── models/               # Trained model files
│   ├── emotion_model.pkl
│   ├── label_encoder.pkl
│   └── scaler.pkl
├── static/
│   └── uploads/          # Temporary file storage
└── templates/
    └── index.html        # Web interface
```

### 5. Train or Load Models
```bash
# If you have training data
python model_trainer.py

# Or download pre-trained models
# Place model files in the models/ directory
```

## 🎯 Usage

### Starting the Application
```bash
python app.py
```
The application will be available at `http://localhost:5000`

### Using the Web Interface

1. **Upload Audio File**
   - Click "Choose File" or drag & drop
   - Supported formats: WAV, MP3, FLAC, M4A, OGG
   - Maximum file size: 16MB

2. **View Results**
   - Predicted emotion with confidence score
   - Probability distribution for all emotions
   - Interactive bar and pie charts

3. **Interpret Results**
   - Higher percentages indicate stronger emotional presence
   - Multiple emotions may be detected simultaneously

### API Usage

#### Upload and Analyze Audio
```bash
curl -X POST http://localhost:5000/upload \
  -F "file=@path/to/your/audio.wav"
```

#### Response Format
```json
{
  "predicted_emotion": "happy",
  "probabilities": {
    "angry": 0.15,
    "disgust": 0.05,
    "fear": 0.10,
    "happy": 0.60,
    "neutral": 0.05,
    "sad": 0.03,
    "surprise": 0.02
  },
  "chart_url": "data:image/png;base64,..."
}
```

## 🧠 Machine Learning Pipeline

### Feature Extraction
- **MFCC (Mel-frequency cepstral coefficients)**
- **Spectral features** (centroid, bandwidth, rolloff)
- **Zero crossing rate**
- **Chroma features**
- **Tempo and rhythm features**

### Model Architecture
- **Algorithm**: Ensemble classifier (Random Forest/SVM)
- **Input**: 40+ audio features
- **Output**: 7 emotion categories
- **Preprocessing**: StandardScaler normalization

### Supported Emotions
- 😠 **Angry**
- 🤢 **Disgust**
- 😨 **Fear**
- 😊 **Happy**
- 😐 **Neutral**
- 😢 **Sad**
- 😲 **Surprise**

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 85.2% |
| Precision | 84.7% |
| Recall | 85.1% |
| F1-Score | 84.9% |

*Performance may vary based on audio quality and training data*

## 🔧 Configuration

### Environment Variables
```bash
# Optional configuration
export FLASK_ENV=development
export FLASK_DEBUG=1
export MAX_CONTENT_LENGTH=16777216  # 16MB
```

### Application Settings
```python
# In app.py
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
```

## 🛠️ Development

### Project Structure
```
├── app.py                 # Flask application and routes
├── feature_extractor.py   # Audio processing utilities
├── model_trainer.py       # ML model training pipeline
└── EmotionRecognizer     # Main prediction class
```

### Adding New Features

1. **New Emotion Categories**
   - Update training data with new labels
   - Retrain model using `model_trainer.py`
   - Update visualization colors

2. **Additional Audio Features**
   - Modify `AudioFeatureExtractor` class
   - Retrain model with expanded feature set

3. **New Audio Formats**
   - Add format to `ALLOWED_EXTENSIONS`
   - Test compatibility with librosa

### Testing
```bash
# Run unit tests
python -m pytest tests/

# Test API endpoints
python -m pytest tests/test_api.py

# Test model accuracy
python -m pytest tests/test_model.py
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Files Not Found**
   ```
   Error: Model files not found. Please train the model first.
   Solution: Run model_trainer.py or download pre-trained models
   ```

2. **Audio Processing Errors**
   ```
   Error: Could not process audio file
   Solution: Check file format and ensure audio is not corrupted
   ```

3. **Memory Issues**
   ```
   Error: Memory allocation failed
   Solution: Reduce file size or increase system RAM
   ```

4. **Dependencies Missing**
   ```bash
   pip install librosa soundfile
   # On Ubuntu/Debian
   sudo apt-get install libsndfile1
   ```

### Performance Optimization

- **Large Files**: Consider audio compression
- **Batch Processing**: Process multiple files sequentially
- **Memory Management**: Automatic file cleanup implemented
- **Caching**: Model loading cached in memory

## 📈 Monitoring & Logging

### Application Logs
```python
import logging
logging.basicConfig(level=logging.INFO)
```

### Performance Metrics
- File processing time
- Model prediction latency
- Memory usage monitoring
- Error rate tracking

## 🔒 Security Features

- **File Type Validation**: Only allowed audio formats
- **Secure Filename Handling**: Prevents directory traversal
- **File Size Limits**: 16MB maximum upload
- **Automatic Cleanup**: Temporary files removed after processing
- **Input Sanitization**: All inputs validated

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Test with various audio formats


