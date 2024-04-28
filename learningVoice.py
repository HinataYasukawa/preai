import numpy as np
import librosa
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    S = np.abs(librosa.stft(y))
    mel_spec = librosa.feature.melspectrogram(S=S, sr=sr)
    dB = librosa.power_to_db(mel_spec, ref=np.max)
    silence_threshold = -60
    silent_frames = (dB < silence_threshold).all(axis=0)
    silence_ratio = np.sum(silent_frames) / len(silent_frames)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[magnitudes > np.median(magnitudes)]
    if len(pitches) > 0:
        pitch_mean = np.mean(pitches)
        pitch_deviation = np.mean(np.abs(pitches - pitch_mean))
    else:
        pitch_deviation = 0

    return silence_ratio, pitch_deviation

def load_data_and_labels(audio_folder, labels):
    features = []
    labels_array = []
    for filename in os.listdir(audio_folder):
        if filename.endswith('.wav'):
            file_path = os.path.join(audio_folder, filename)
            label = labels.get(filename, None)
            if label is not None:
                silence_ratio, pitch_deviation = extract_features(file_path)
                features.append([silence_ratio, pitch_deviation])
                labels_array.append(label)
    return np.array(features), np.array(labels_array)

def main():
    audio_folder = "C:/openpose/examples/audio/audio.mp3"
    labels = {
        'audio1.mp3': "good",
    }

    features, labels = load_data_and_labels(audio_folder, labels)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    print(classification_report(y_test, predictions))

if __name__ == '__main__':
    main()
