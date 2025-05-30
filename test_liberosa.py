import librosa
import numpy as np
import os

print(f"Librosa version: {librosa.__version__}")

# Test basic audio loading
test_file = None
data_path = 'data'

# Find a test audio file
for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith('.wav'):
            test_file = os.path.join(root, file)
            break
    if test_file:
        break

if test_file:
    print(f"Testing with file: {test_file}")
    
    try:
        # Test audio loading
        audio, sr = librosa.load(test_file, duration=3, offset=0.5)
        print(f"Audio loaded successfully. Length: {len(audio)}, Sample rate: {sr}")
        
        # Test MFCC
        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            print(f"MFCC extraction successful. Shape: {mfcc.shape}")
        except Exception as e:
            print(f"MFCC failed: {e}")
        
        # Test chroma - try different methods
        try:
            if hasattr(librosa.feature, 'chroma_stft'):
                chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
                print(f"Chroma (stft) extraction successful. Shape: {chroma.shape}")
            elif hasattr(librosa.feature, 'chroma'):
                chroma = librosa.feature.chroma(y=audio, sr=sr)
                print(f"Chroma extraction successful. Shape: {chroma.shape}")
            else:
                print("No chroma function found")
                print("Available librosa.feature attributes:")
                print([attr for attr in dir(librosa.feature) if 'chroma' in attr.lower()])
        except Exception as e:
            print(f"Chroma failed: {e}")
            
        # Test other features
        try:
            zcr = librosa.feature.zero_crossing_rate(audio)
            print(f"ZCR extraction successful. Shape: {zcr.shape}")
        except Exception as e:
            print(f"ZCR failed: {e}")
            
        try:
            spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
            print(f"Spectral centroid extraction successful. Shape: {spec_cent.shape}")
        except Exception as e:
            print(f"Spectral centroid failed: {e}")
            
    except Exception as e:
        print(f"Error loading audio file: {e}")
else:
    print("No audio files found in data directory")
    print("Please make sure RAVDESS dataset is in the 'data' folder")

# List all available librosa.feature functions
print("\nAll available librosa.feature functions:")
feature_functions = [attr for attr in dir(librosa.feature) if not attr.startswith('_')]
for func in sorted(feature_functions):
    print(f"  - {func}")