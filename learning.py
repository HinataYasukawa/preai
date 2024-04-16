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

def load_and_process_data(json_dir, labels, video_name):
    features = []
    label_data = []
    previous_keypoints = None

    full_json_dir = os.path.join(os.getcwd(), json_dir)
    print(f"Looking for JSON files in: {full_json_dir}")

    if not os.listdir(full_json_dir):
        print(f"No files in {full_json_dir}")
        return np.array(features), np.array(label_data)

    for filename in sorted(os.listdir(json_dir)):
        file_path = os.path.join(json_dir, filename)
        if not file_path.endswith('.json'):
            print(f"Skipped non-JSON file: {filename}")
            continue

        print(f"Processing file: {filename}")
        if video_name not in labels:
            print(f"No label for video: {video_name}")
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)

        # データに人がいない場合スキップ
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
                delta_keypoints = np.array(current_keypoints) - np.array(previous_keypoints)
                features.append(delta_keypoints)
                label_data.append(labels[video_name])

            previous_keypoints = current_keypoints

    print(f"Loaded {len(features)} feature sets.")
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
video_name = '01.mp4'
label_file = 'labels.json'
labels = load_labels(label_file)
json_dir = 'output'
features, label_data = load_and_process_data(json_dir, labels, video_name)
train_model(features, label_data)
