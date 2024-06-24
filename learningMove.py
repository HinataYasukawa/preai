import numpy as np
import json
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

def load_labels(label_file):
    with open(label_file, 'r') as file:
        return json.load(file)

def load_and_process_data(json_dir, labels, video_name):
    features = []
    label_data = []
    previous_keypoints = None
    deltaf = []
    n = 0

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

        if video_name not in labels:
            print(f"No label for video: {video_name}")
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

    if n > 0:
        average = np.mean(deltaf, axis=0)
    else:
        average = np.zeros_like(current_keypoints)

    print(n)
    print(average)
    features.append(average)
    label_data.append(labels[video_name])

    print(f"Loaded {len(features)} feature sets.")
    return np.array(features), np.array(label_data)

def train_model(features, labels, model_file='model1.pkl'):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, labels)
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")

# 実行部分
video_name = "30.mp4"
label_file = 'labels.json'
labels = load_labels(label_file)
json_dir = 'output/json'
features, label_data = load_and_process_data(json_dir, labels, video_name)
train_model(features, label_data)
