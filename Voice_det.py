import librosa
import numpy as np

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    
    # Extracting MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    # Extracting Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Extracting Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_mean = np.mean(mel, axis=1)
    
    # Concatenate all features
    features = np.concatenate((mfccs_mean, chroma_mean, mel_mean))
    
    return features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Load dataset

def load_dataset(dataset_path):
    features = []
    labels = []
    for file in os.listdir(dataset_path):
        if file.endswith(".wav"):
            print(f"Processing file: {file}")  # Debug statement
            emotion = file.split("_")[-1].split(".")[0]  # Adjust this line based on your filename format
            feature = extract_features(os.path.join(dataset_path, file))
            features.append(feature)
            labels.append(emotion)
    print(f"Total files processed: {len(features)}")
    return np.array(features), np.array(labels)

dataset_path = 'C:/Users/KUMARAN/OneDrive/Desktop/dataset/TESSTorontoemotionalspeechsetdata'
features, labels = load_dataset(dataset_path)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
import pyaudio
import wave

def record_audio(filename, duration=5, fs=44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)
    
    print("Recording...")
    frames = []
    for _ in range(0, int(fs / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    print("Done recording.")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

def predict_emotion(audio_path):
    features = extract_features(audio_path)
    features = features.reshape(1, -1)
    emotion = clf.predict(features)
    return emotion[0]

def main():
    while True:
        record_audio('live_audio.wav', duration=20)
        emotion = predict_emotion('live_audio.wav')
        print(f'Predicted Emotion: {emotion}')
        
        if input("Continue? (y/n): ").lower() != 'y':
            break

if __name__ == "__main__":
    main()
