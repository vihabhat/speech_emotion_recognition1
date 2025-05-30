from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import numpy as np
import joblib
from werkzeug.utils import secure_filename
from feature_extractor import AudioFeatureExtractor
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class EmotionRecognizer:
    def __init__(self):
        self.feature_extractor = AudioFeatureExtractor()
        self.load_model()
    
    def load_model(self):
        """Load trained model components"""
        try:
            self.model = joblib.load('models/emotion_model.pkl')
            self.label_encoder = joblib.load('models/label_encoder.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("Model files not found. Please train the model first using model_trainer.py")
            self.model = None
    
    def predict_emotion(self, audio_file_path):
        """Predict emotion from audio file"""
        if self.model is None:
            return None, None
        
        # Extract features
        features = self.feature_extractor.extract_features(audio_file_path)
        if features is None:
            return None, None
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get emotion labels and probabilities
        emotions = self.label_encoder.classes_
        emotion_probs = dict(zip(emotions, probabilities))
        
        # Get predicted emotion
        predicted_label = self.model.predict(features_scaled)[0]
        predicted_emotion = self.label_encoder.inverse_transform([predicted_label])[0]
        
        return predicted_emotion, emotion_probs

# Initialize emotion recognizer
recognizer = EmotionRecognizer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict emotion
        predicted_emotion, emotion_probs = recognizer.predict_emotion(filepath)
        
        if predicted_emotion is None:
            return jsonify({'error': 'Could not process audio file'})
        
        # Generate visualization
        chart_url = generate_emotion_chart(emotion_probs)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'predicted_emotion': predicted_emotion,
            'probabilities': emotion_probs,
            'chart_url': chart_url
        })
    
    return jsonify({'error': 'Invalid file type'})

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_emotion_chart(emotion_probs):
    """Generate emotion probability chart"""
    emotions = list(emotion_probs.keys())
    probabilities = [prob * 100 for prob in emotion_probs.values()]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart
    colors = plt.cm.viridis(np.linspace(0, 1, len(emotions)))
    bars = ax1.bar(emotions, probabilities, color=colors)
    ax1.set_title('Emotion Recognition Results', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Probability (%)', fontsize=12)
    ax1.set_xlabel('Emotions', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add percentage labels on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(emotions)))
    wedges, texts, autotexts = ax2.pie(probabilities, labels=emotions, autopct='%1.1f%%',
                                      colors=colors_pie, startangle=90)
    ax2.set_title('Emotion Distribution', fontsize=16, fontweight='bold')
    
    # Enhance pie chart text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plot_data = buffer.getvalue()
    buffer.close()
    plt.close()
    
    plot_url = base64.b64encode(plot_data).decode()
    return f"data:image/png;base64,{plot_url}"

if __name__ == '__main__':
    app.run(debug=True)