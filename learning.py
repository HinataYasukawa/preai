import numpy as np
import json
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_labels(label_file):
    with open(label_file, 'r') as file:
        return json.load(file)

def load_and_process_data(json_dir, labels):
    features = []
    label_data = []  # 実際のラベルを格納するリスト
    previous_keypoints = None

    for filename in sorted(os.listdir(json_dir)):
        file_path = os.path.join(json_dir, filename)
        if not file_path.endswith('.json'):
            continue

        video_name = filename.split('_')[0] + '.mp4'
        if video_name not in labels:
            continue  # ラベルが存在しない動画はスキップ

        with open(file_path, 'r') as f:
            data = json.load(f)

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
                delta_keypoints = np.array(current_keypoints) - np.array(previous_keypoints)
                features.append(delta_keypoints)
                label_data.append(labels[video_name])  # 対応するラベルを追加

            previous_keypoints = current_keypoints

    return np.array(features), np.array(label_data)

def train_model(features, labels, model_file='model.pkl'):
    if os.path.exists(model_file):
        model = joblib.load(model_file)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    joblib.dump(model, model_file)

# 実行部分
label_file = 'labels.json'
labels = load_labels(label_file)
json_dir = 'output'
features, label_data = load_and_process_data(json_dir, labels)
train_model(features, label_data)
