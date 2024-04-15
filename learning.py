import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 特徴量とラベルの準備
features = []
targets = []  # このリストには手動でラベル（good, normal, bad）を追加する必要があります

# 前のフレームのキーポイントを保持するための変数
previous_keypoints = None

# 10フレームのデータを読み込む
for num in range(600):
    n = f'{num:04}'
    with open('frame_' + n + '_keypoints.json', 'r') as json_open:
        data = json.load(json_open)

        indices = [0, 4, 7]
        keypoint_names = ['face', 'right', 'left']

        for person in data['people']:
            keypoints = person['pose_keypoints_2d']

            # 現在のフレームのキーポイントを初期化
            current_keypoints = []

            for index in indices:
                x = keypoints[index * 3]  # x座標
                y = keypoints[index * 3 + 1]  # y座標
                confidence = keypoints[index * 3 + 2]  # 信頼度
                current_keypoints.extend([x, y, confidence])

            if previous_keypoints is not None:
                # 前のフレームとの差分を計算
                delta_keypoints = np.array(current_keypoints) - np.array(previous_keypoints)
                features.append(delta_keypoints)  # 特徴量リストに追加

            # 現在のキーポイントを更新
            previous_keypoints = current_keypoints

# 特徴量とターゲットをNumpy配列に変換
X = np.array(features)
Y = np.array(targets)  # 対応するターゲットラベルを用意

# データを訓練セットとテストセットに分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 分類器の訓練
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

# モデルの評価
predictions = model.predict(X_test)
print(classification_report(Y_test, predictions))