import numpy as np
import librosa
import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier

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

def load_labels(label_file):
    with open(label_file, 'r') as file:
        return json.load(file)

#labels.jsonの末尾の.mp4のラベルをロード
def load_data_and_labels(audio_folder, labels, filename):
    features = []
    labels_array = []
    for filename in os.listdir(audio_folder):
        if filename.endswith('.wav'):
            file_path = os.path.join(audio_folder, filename)
            video_name = os.path.splitext(filename)[0] + ".mp4"
            print(video_name)
            label = labels.get(video_name)
            if label is not None:
                silence_ratio, pitch_deviation = extract_features(file_path)
                features.append([silence_ratio, pitch_deviation])
                labels_array.append(label)
    return np.array(features), np.array(labels_array)

def main():
    audio_folder = "C:/openpose/output/audio/"
    label_file = 'labels.json'
    labels = load_labels(label_file)
    filename = "audio.wav"

    features, labels_array = load_data_and_labels(audio_folder, labels, filename)
    if len(features) == 0:
        print("No data available for training.")
        return

    model_path = 'model2.pkl'
    if os.path.exists(model_path):
        classifier = joblib.load(model_path)
        print("Loaded existing model.")
    else:
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        print("Created new model.")

    classifier.fit(features, labels_array)
    print("Model trained on all available data.")

    joblib.dump(classifier, model_path)
    print("Model saved to", model_path)

if __name__ == '__main__':
    main()
