import subprocess
import os
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from nltk.sentiment import SentimentIntensityAnalyzer
import speech_recognition as sr

# main.pyを実行して動画の処理を行う
def run_main_py(video_path):
    command = f"python main.py --video_path {video_path}"
    subprocess.run(command.split(), check=True)

# JSONディレクトリからポーズのキーポイント特徴量を読み込む
def load_point_features(json_dir):
    features = []
    for filename in sorted(os.listdir(json_dir)):
        if filename.endswith('.json'):
            file_path = os.path.join(json_dir, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                keypoints = data['people'][0]['pose_keypoints_2d']
                features.append(keypoints)
    return np.array(features)

# 音声ファイルから声の特徴量を解析する
def load_voice_features(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio, language='ja-JP')
        sia = SentimentIntensityAnalyzer()
        sentiment_score = sia.polarity_scores(text)['compound']
        return sentiment_score
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return 0
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {e}")
        return 0

# テキストファイルからテキストの特徴量を解析する
def load_text_features(text_file):
    with open(text_file, 'r', encoding='utf-8') as file:
        text = file.read()
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    text_length = len(text.split())
    return np.array([sentiment_score, text_length])

# 入力動画に対してフィードバックを生成する
def generate_feedback(video_path, model_paths, json_dir='output', audio_path='output_audio.wav', text_path='output_text.txt'):
    # main.pyを実行して動画からデータを抽出
    run_main_py(video_path)

    # 特徴量の読み込み
    point_features = load_point_features(json_dir)
    voice_feature = load_voice_features(audio_path)
    text_features = load_text_features(text_path)

    # モデルのロードと予測
    point_model = joblib.load(model_paths['pose'])
    voice_model = joblib.load(model_paths['audio'])
    text_model = joblib.load(model_paths['text'])

    point_score = point_model.predict(point_features).mean()
    voice_score = voice_model.predict(np.array([[voice_feature]]))[0]
    text_score = text_model.predict(text_features)[0]

    # 統合スコアの計算
    integrated_score = (point_score + voice_score + text_score) / 3
    return f"このプレゼンテーションのスコアは {integrated_score:.2f}/10 です。"

if __name__ == "__main__":
    video_path = 'openpose/input/your_presentation_video.mp4'
    model_paths = {
        'pose': 'model_pose.pkl',
        'audio': 'model_audio.pkl',
        'text': 'model_text.pkl'
    }
    feedback = generate_feedback(video_path, model_paths)
    print(feedback)
