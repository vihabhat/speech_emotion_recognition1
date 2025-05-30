import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AudioFeatureExtractor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_features(self, file_path, duration=3):
        """
        Extract audio features from a given audio file
        """
        try:
            # Load audio file
            audio, sample_rate = librosa.load(file_path, duration=duration, offset=0.5)
            
            # Ensure audio is not empty
            if len(audio) == 0:
                print(f"Empty audio file: {file_path}")
                return None
            
            # Extract features
            features = {}
            
            # 1. MFCC (Mel-frequency cepstral coefficients)
            try:
                mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
                features['mfcc_mean'] = np.mean(mfcc, axis=1)
                features['mfcc_std'] = np.std(mfcc, axis=1)
            except Exception as e:
                print(f"MFCC extraction failed for {file_path}: {e}")
                # Use zeros as fallback
                features['mfcc_mean'] = np.zeros(13)
                features['mfcc_std'] = np.zeros(13)
            
            # 2. Chroma features (with fallback for different librosa versions)
            try:
                # Try different ways to access chroma
                if hasattr(librosa.feature, 'chroma_stft'):
                    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
                elif hasattr(librosa.feature, 'chroma'):
                    chroma = librosa.feature.chroma(y=audio, sr=sample_rate)
                else:
                    # Fallback: compute chroma manually
                    chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
                
                features['chroma_mean'] = np.mean(chroma, axis=1)
                features['chroma_std'] = np.std(chroma, axis=1)
            except Exception as e:
                print(f"Chroma extraction failed for {file_path}: {e}")
                # Use zeros as fallback
                features['chroma_mean'] = np.zeros(12)
                features['chroma_std'] = np.zeros(12)
            
            # 3. Spectral centroid
            try:
                spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
                features['spectral_centroid_mean'] = np.mean(spectral_centroid)
                features['spectral_centroid_std'] = np.std(spectral_centroid)
            except Exception as e:
                print(f"Spectral centroid extraction failed for {file_path}: {e}")
                features['spectral_centroid_mean'] = 0.0
                features['spectral_centroid_std'] = 0.0
            
            # 4. Zero crossing rate
            try:
                zcr = librosa.feature.zero_crossing_rate(audio)
                features['zcr_mean'] = np.mean(zcr)
                features['zcr_std'] = np.std(zcr)
            except Exception as e:
                print(f"ZCR extraction failed for {file_path}: {e}")
                features['zcr_mean'] = 0.0
                features['zcr_std'] = 0.0
            
            # 5. Spectral rolloff
            try:
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
                features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
                features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            except Exception as e:
                print(f"Spectral rolloff extraction failed for {file_path}: {e}")
                features['spectral_rolloff_mean'] = 0.0
                features['spectral_rolloff_std'] = 0.0
            
            # 6. RMS Energy
            try:
                rms = librosa.feature.rms(y=audio)
                features['rms_mean'] = np.mean(rms)
                features['rms_std'] = np.std(rms)
            except Exception as e:
                print(f"RMS extraction failed for {file_path}: {e}")
                features['rms_mean'] = 0.0
                features['rms_std'] = 0.0
            
            # 7. Spectral bandwidth
            try:
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
                features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
                features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
            except Exception as e:
                print(f"Spectral bandwidth extraction failed for {file_path}: {e}")
                features['spectral_bandwidth_mean'] = 0.0
                features['spectral_bandwidth_std'] = 0.0
            
            # 8. Tempo (simplified approach)
            try:
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
                features['tempo'] = tempo if not np.isnan(tempo) else 120.0
            except Exception as e:
                print(f"Tempo extraction failed for {file_path}: {e}")
                features['tempo'] = 120.0  # Default tempo
            
            # Flatten all features into a single array
            feature_vector = []
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    feature_vector.extend(value.flatten())
                else:
                    feature_vector.append(float(value))
            
            # Ensure we have a consistent feature vector length
            feature_vector = np.array(feature_vector)
            
            # Handle any NaN or infinite values
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
                    
            return feature_vector
            
        except Exception as e:
            print(f"Error extracting features from {file_path}: {e}")
            return None
    
    def extract_dataset_features(self, file_paths, labels):
        """
        Extract features from multiple audio files
        """
        features_list = []
        valid_labels = []
        
        print(f"Processing {len(file_paths)} audio files...")
        
        for i, file_path in enumerate(file_paths):
            if i % 50 == 0:  # Progress indicator
                print(f"Processing file {i+1}/{len(file_paths)}")
                
            features = self.extract_features(file_path)
            if features is not None and len(features) > 0:
                features_list.append(features)
                valid_labels.append(labels[i])
            else:
                print(f"Skipping file {file_path} - no features extracted")
                
        if len(features_list) == 0:
            print("No valid features extracted from any files!")
            return np.array([]), np.array([])
            
        print(f"Successfully extracted features from {len(features_list)} files")
        return np.array(features_list), np.array(valid_labels)