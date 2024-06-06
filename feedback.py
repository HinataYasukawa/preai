import os
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from nltk.sentiment import SentimentIntensityAnalyzer
import librosa

# JSONディレクトリからポーズのキーポイント特徴量を読み込む
def load_point_features(json_dir):
    features = []
    previous_keypoints = None
    for filename in sorted(os.listdir(json_dir)):
        file_path = os.path.join(json_dir, filename)
        if not file_path.endswith('.json'):
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
                current_keypoints.extend([x, y, confidence])

            if previous_keypoints is not None:
                delta_keypoints = np.array(current_keypoints, dtype=np.float32) - np.array(previous_keypoints, dtype=np.float32)
                features.append(delta_keypoints)

            previous_keypoints = current_keypoints

    return np.array(features, dtype=np.float32)

# 音声ファイルから声の特徴量を解析する
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

# テキストファイルからテキストの特徴量を解析する
def load_text_features(text_file):
    with open(text_file, 'r', encoding='utf-8') as file:
        text = file.read()

    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    text_length = len(text.split())

    return np.array([sentiment_score, text_length], dtype=np.float32)

# ラベルを数値に変換
def convert_label_to_score(label):
    if label == "good":
        return 3
    elif label == "normal":
        return 2
    elif label == "bad":
        return 1
    else:
        return 0

# 過去の動画の特長量を読み込む
def load_past_features(feature_dir):
    point_features = np.load(os.path.join(feature_dir, 'point_features.npy'), allow_pickle=True)
    voice_features = np.load(os.path.join(feature_dir, 'voice_features.npy'), allow_pickle=True)
    text_features = np.load(os.path.join(feature_dir, 'text_features.npy'), allow_pickle=True)

    return point_features, voice_features, text_features

# 新しい動画の特長量を過去の動画の特長量と比較してフィードバックを生成する
def generate_feedback(video_name, model_paths, feature_dir, json_dir="output/json", audio_dir="output/audio", txt_dir="output/txt"):
    # ディレクトリとファイルのパス設定
    audio_path = os.path.join(audio_dir, video_name.replace('.mp4', '.wav'))
    txt_path = os.path.join(txt_dir, video_name.replace('.mp4', '.txt'))

    # 新しい動画の特長量の読み込み
    new_point_features = load_point_features(json_dir)
    new_voice_features = load_voice_features(audio_path)
    new_text_features = load_text_features(txt_path)

    # 過去の動画の特長量の読み込み
    past_point_features, past_voice_features, past_text_features = load_past_features(feature_dir)

    # モデルのロードと予測
    point_model = joblib.load(model_paths['pose'])
    voice_model = joblib.load(model_paths['audio'])
    text_model = joblib.load(model_paths['text'])

    new_point_predictions = point_model.predict(new_point_features)
    new_voice_prediction = voice_model.predict([new_voice_features])[0]
    new_text_prediction = text_model.predict([new_text_features])[0]

    # 予測結果のスコア変換
    new_point_score = np.mean([convert_label_to_score(pred) for pred in new_point_predictions])
    new_voice_score = convert_label_to_score(new_voice_prediction)
    new_text_score = convert_label_to_score(new_text_prediction)

    # 過去の動画のスコア計算
    past_point_scores = np.mean([convert_label_to_score(pred) for preds in past_point_features for pred in preds])
    past_voice_scores = np.mean([convert_label_to_score(pred) for pred in past_voice_features])
    past_text_scores = np.mean([convert_label_to_score(pred) for pred in past_text_features])

    # フィードバック生成
    feedback = f"このプレゼンテーションのスコアは {new_point_score:.2f}/10 です。\n"
    feedback += f"動き: 新しいスコア {new_point_score:.2f}, 過去の平均スコア {past_point_scores:.2f}\n"
    feedback += f"声の特徴: 新しいスコア {new_voice_score:.2f}, 過去の平均スコア {past_voice_scores:.2f}\n"
    feedback += f"テキストの特徴: 新しいスコア {new_text_score:.2f}, 過去の平均スコア {past_text_scores:.2f}\n"

    return feedback

if __name__ == "__main__":
    video_name = '32.mp4'
    model_paths = {
        'pose': 'model1.pkl',
        'audio': 'model2.pkl',
        'text': 'model3.pkl'
    }
    feature_dir = "output/features"
    feedback = generate_feedback(video_name, model_paths, feature_dir)
    print(feedback)
