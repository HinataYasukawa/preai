import subprocess
import os
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

    #main.pyを実行して動画の処理を行います。
def run_main_py(video_path):
    command = f"python main.py --video_path {video_path}"
    subprocess.run(command.split(), check=True)


    #JSONディレクトリから特徴量を読み込みます。
def load_features(json_dir):
    features = []
    for filename in sorted(os.listdir(json_dir)):
        file_path = os.path.join(json_dir, filename)
        if not file_path.endswith('.json'):
            continue
        with open(file_path, 'r') as f:
            data = json.load(f)
            keypoints = data['people'][0]['pose_keypoints_2d']
            features.append(keypoints)
    return np.array(features)


    #入力動画に対してフィードバックを生成
def generate_feedback(video_path, model_path='model.pkl', json_dir='output'):

    # main.pyを実行して動画からキーポイントを抽出
    run_main_py(video_path)

    # 特徴量の読み込み
    features = load_features(json_dir)

    # モデルのロード
    model = joblib.load(model_path)

    # 特徴量から点数を予測
    predictions = model.predict(features)
    score = np.mean(predictions)
    score = (score / len(np.unique(predictions))) * 10

    return f"このプレゼンテーションのスコアは {score:.2f}/10 です。"


if __name__ == "__main__":
    video_path = 'openpose/input/your_presentation_video.mp4'
    feedback = generate_feedback(video_path)
    print(feedback)
