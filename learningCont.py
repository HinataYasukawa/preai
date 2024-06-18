import nltk
import os
import json
import joblib
import numpy as np
import Mecab
import ipadic
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier

nltk.download('vader_lexicon')

def load_labels(label_file):
    with open(label_file, 'r') as file:
        return json.load(file)

def load_text_features(text_file):
    # テキストファイルから感情スコアとテキストの長さを特徴量として抽出
    with open(text_file, 'r', encoding='utf-8') as file:
        text = file.read()

    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    text_length = len(text.split())

    mecab = Mecab.Tagger()

    return [sentiment_score, text_length]

def load_data_and_labels(text_folder, labels):
    features = []
    labels_array = []
    for filename in os.listdir(text_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(text_folder, filename)
            video_name = os.path.splitext(filename)[0] + ".mp4"
            label = labels.get(video_name)
            if label is not None:
                text_features = load_text_features(file_path)
                features.append(text_features)
                labels_array.append(label)
    return np.array(features), np.array(labels_array)

def main():
    text_folder = "C:/openpose/output/txt/"
    label_file = 'labels.json'
    labels = load_labels(label_file)

    features, labels_array = load_data_and_labels(text_folder, labels)
    if len(features) == 0:
        print("No data available for training.")
        return

    model_path = 'model3.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Loaded existing model.")
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        print("Created new model.")

    # 全データを使用してモデルを訓練
    model.fit(features, labels_array)
    print("Model trained on all available data.")

    # モデルを保存
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    main()
