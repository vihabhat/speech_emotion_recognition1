import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from feature_extractor import AudioFeatureExtractor
import seaborn as sns
import matplotlib.pyplot as plt

class EmotionModelTrainer:
    def __init__(self):
        self.feature_extractor = AudioFeatureExtractor()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # RAVDESS emotion mapping
        self.emotion_mapping = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
    
    def load_ravdess_data(self, data_path='data'):
        """
        Load RAVDESS dataset and extract file paths and labels
        """
        file_paths = []
        labels = []
        
        # Pattern: 03-01-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav
        pattern = os.path.join(data_path, "Actor_*", "*.wav")
        audio_files = glob.glob(pattern)
        
        for file_path in audio_files:
            filename = os.path.basename(file_path)
            parts = filename.split('-')
            
            if len(parts) >= 3:
                emotion_code = parts[2]
                if emotion_code in self.emotion_mapping:
                    file_paths.append(file_path)
                    labels.append(self.emotion_mapping[emotion_code])
        
        return file_paths, labels
    
    def train_model(self):
        """
        Train the emotion recognition model
        """
        print("Loading RAVDESS dataset...")
        file_paths, labels = self.load_ravdess_data()
        
        if len(file_paths) == 0:
            raise ValueError("No audio files found. Please check if RAVDESS dataset is properly placed in 'data/' folder")
        
        print(f"Found {len(file_paths)} audio files")
        print("Extracting features...")
        
        # Extract features
        X, y = self.feature_extractor.extract_dataset_features(file_paths, labels)
        
        if len(X) == 0:
            raise ValueError("No features extracted. Please check audio files.")
        
        print(f"Extracted features from {len(X)} files")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        # Save model components
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/emotion_model.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        print("\nModel saved to 'models/' directory")
        
        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
        return self.model, self.label_encoder, self.scaler
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Emotion Recognition Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png')
        plt.show()

if __name__ == "__main__":
    trainer = EmotionModelTrainer()
    trainer.train_model()