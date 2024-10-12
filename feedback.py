import os
import json
import numpy as np
import joblib
from janome.tokenizer import Tokenizer
from sklearn.ensemble import RandomForestClassifier
from nltk.sentiment import SentimentIntensityAnalyzer
import librosa

def count_pos(text):
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(text)
    
    pos_counter = [0, 0]

    for token in tokens:
        pos = token.part_of_speech.split(',')[0]
        if pos == "フィラー":
            pos_counter[0] += 1
        if pos == "感動詞":
            pos_counter[1] += 1

    return pos_counter

def load_point_features(json_dir):
    features = []
    previous_keypoints = None
    deltaf = []
    n = 0

    full_json_dir = os.path.join(os.getcwd(), json_dir)
    print(f"Looking for JSON files in: {full_json_dir}")

    if not os.listdir(full_json_dir):
        print(f"No files in {full_json_dir}")
        return np.array(features)

    for filename in sorted(os.listdir(full_json_dir)):
        file_path = os.path.join(full_json_dir, filename)
        if not file_path.endswith('.json'):
            print(f"Skipped non-JSON file: {filename}")
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)

        if not data['people']:
            continue

        for person in data['people']:
            keypoints = person['pose_keypoints_2d']
            indices = [0, 4, 7]
            current_keypoints = []

            for index in indices:
                x = keypoints[index * 3]
                y = keypoints[index * 3 + 1]
                confidence = keypoints[index * 3 + 2]
                key = (x + y) * confidence
                current_keypoints.append(key)

            if previous_keypoints is not None:
                diffs = [current - previous for current, previous in zip(current_keypoints, previous_keypoints)]
                if any(abs(diff) >= 300 for diff in diffs):
                    delta_keypoints = np.array(current_keypoints) - np.array(previous_keypoints)
                    deltaf.append(delta_keypoints)
                    n += 1

            previous_keypoints = current_keypoints

    average = np.mean(deltaf, axis=0) if deltaf else np.zeros(3)
    print(n)
    print(average)
    features.append(average)

    print(f"Loaded {len(features)} feature sets.")
    return np.array(features)

def load_voice_features(audio_path):
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

    return np.array([silence_ratio, pitch_deviation], dtype=np.float32)

def load_text_features(text_file):
    with open(text_file, 'r', encoding='utf-8') as file:
        text = file.read()

    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    text_length = len(text.split())

    pos_counts = count_pos(text)

    tmpf = [sentiment_score, text_length]
    score = tmpf + pos_counts
    print(score)

    return np.array(score, dtype=np.float32)

def load_past_features(feature_dir):
    point_features = np.load(os.path.join(feature_dir, 'point_features.npy'), allow_pickle=True)
    voice_features = np.load(os.path.join(feature_dir, 'voice_features.npy'), allow_pickle=True)
    text_features = np.load(os.path.join(feature_dir, 'text_features.npy'), allow_pickle=True)

    return point_features, voice_features, text_features

def generate_feedback(name, model_paths, feature_dir):
    json_dir = os.path.join("test/json/", name)
    audio_path = os.path.join("test/audio/", name + ".wav")
    txt_path = os.path.join("test/txt/", name + ".txt")

    new_point_features = load_point_features(json_dir)
    new_voice_features = load_voice_features(audio_path)
    new_text_features = load_text_features(txt_path)
    print(new_voice_features)
    print("new text features:")
    print(new_text_features)

    point_model = joblib.load(model_paths['pose'])
    voice_model = joblib.load(model_paths['audio'])
    text_model = joblib.load(model_paths['text'])

    new_point_predictions = point_model.predict(new_point_features)
    new_voice_prediction = voice_model.predict([new_voice_features])
    new_text_prediction = text_model.predict([new_text_features])
    print(new_text_prediction)

    new_point_score = np.mean(new_point_predictions)
    new_voice_score = new_voice_prediction[0]
    new_text_score = new_text_prediction[0]
    all_score = (new_point_score + new_voice_score + new_text_score) / 3
    print("new scores:")
    print(new_point_score)
    print(new_voice_score)
    print(new_text_score)

    past_point_features, past_voice_features, past_text_features = load_past_features(feature_dir)

    past_silence_ratios = [features[0] for features in past_voice_features]
    past_pitch_deviations = [features[1] for features in past_voice_features]
    past_sentiment_scores = [features[0] for features in past_text_features]
    past_text_lengths = [features[1] for features in past_text_features]

    past_silence_ratio_mean = np.mean(past_silence_ratios)
    past_pitch_deviation_mean = np.mean(past_pitch_deviations)
    past_sentiment_score_mean = np.mean(past_sentiment_scores)
    past_text_length_mean = np.mean(past_text_lengths)

    new_silence_ratio = new_voice_features[0]
    new_pitch_deviation = new_voice_features[1]
    new_sentiment_score = new_text_features[0]
    new_text_length = new_text_features[1]

    feedback = "フィードバック:\n"
    
    if new_silence_ratio > past_silence_ratio_mean:
        feedback += "無声空間が多いです。\n"
    else:
        feedback += "無声空間が少ないです。\n"

    if new_pitch_deviation > past_pitch_deviation_mean:
        feedback += "抑揚が多いです。\n"
    else:
        feedback += "抑揚が少ないです。\n"

    if new_sentiment_score > past_sentiment_score_mean:
        feedback += "感情表現が豊かです。\n"
    else:
        feedback += "感情表現が控えめです。\n"

    if new_text_length > past_text_length_mean:
        feedback += "テキストが長いです。\n"
    else:
        feedback += "テキストが短いです。\n"

    feedback += f"動き: スコアは {new_point_score:.2f}です。\n"
    feedback += f"声の特徴: スコアは {new_voice_score:.2f}です。\n"
    feedback += f"テキストの特徴: スコアは {new_text_score:.2f}です。\n"
    feedback += f"このプレゼンテーションのスコアは {all_score:.2f}/3 です。\n"

    return feedback

if __name__ == "__main__":
    name = '001'
    model_paths = {
        'pose': 'model1.pkl',
        'audio': 'model2.pkl',
        'text': 'model3.pkl'
    }
    feature_dir = "output/features"
    feedback = generate_feedback(name, model_paths, feature_dir)
    print(feedback)
