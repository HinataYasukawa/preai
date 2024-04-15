import json
import joblib
import numpy as np

def generate_feedback(video_path):
    model = joblib.load('model.pkl')  # 学習済みモデルの読み込み
