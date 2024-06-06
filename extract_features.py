import os
import numpy as np
import subprocess
from feedback import load_point_features, load_voice_features, load_text_features

def process_video(video_name, video_dir, json_dir, audio_dir, txt_dir):
    # ディレクトリとファイルのパス設定
    video_path = os.path.join(video_dir, video_name)
    audio_path = os.path.join(audio_dir, video_name.replace('.mp4', '.wav'))
    txt_path = os.path.join(txt_dir, video_name.replace('.mp4', '.txt'))

    # 既存のファイルがあるか確認
    if not os.path.exists(audio_path) or not os.path.exists(txt_path) or not any(fname.endswith('.json') for fname in os.listdir(json_dir)):
        number = video_name.split('.')[0]
        command = f"python main.py {number}"
        subprocess.run(command.split(), check=True)

    # 特長量の読み込み
    point_features = load_point_features(json_dir)
    voice_features = load_voice_features(audio_path)
    text_features = load_text_features(txt_path)

    return point_features, voice_features, text_features

def save_features(features, file_path):
    np.save(file_path, features)

def main(video_dir, json_dir, audio_dir, txt_dir, output_dir):
    all_point_features = []
    all_voice_features = []
    all_text_features = []

    # video_dirから.mp4ファイルを取得
    video_list = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

    for video_name in video_list:
        point_features, voice_features, text_features = process_video(video_name, video_dir, json_dir, audio_dir, txt_dir)
        all_point_features.append(point_features)
        all_voice_features.append(voice_features)
        all_text_features.append(text_features)

    save_features(np.array(all_point_features, dtype=object), os.path.join(output_dir, 'point_features.npy'))
    save_features(np.array(all_voice_features, dtype=object), os.path.join(output_dir, 'voice_features.npy'))
    save_features(np.array(all_text_features, dtype=object), os.path.join(output_dir, 'text_features.npy'))

if __name__ == "__main__":
    video_dir = "examples/"
    json_dir = "output/json"
    audio_dir = "output/audio"
    txt_dir = "output/txt"
    output_dir = "output/features"

    os.makedirs(output_dir, exist_ok=True)
    main(video_dir, json_dir, audio_dir, txt_dir, output_dir)
