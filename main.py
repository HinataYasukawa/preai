import subprocess
import cv2
import speech_recognition as sr
import os
import json
from moviepy.editor import VideoFileClip

number = "11"
openpose_path = "bin\OpenPoseDemo.exe"
video_path = "C:/openpose/examples/"+number+".mp4"

output_dir = "output/json"
image_dir = "output/image"
audio_dir = "output/audio"
txt_dir = 'output/txt'

audio_path = os.path.join(audio_dir, number+".wav")
txt_path = os.path.join(txt_dir, number+'.txt')

video = VideoFileClip(video_path)

os.makedirs(audio_dir, exist_ok=True)
video.audio.write_audiofile(audio_path)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

frame_count = 0
image_count = 0

os.makedirs(image_dir, exist_ok=True)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    if frame_count % int(fps) == 0:
        image_path = os.path.join(image_dir, f"frame_{image_count:04d}.jpg")
        cv2.imwrite(image_path, frame)
        image_count += 1

    frame_count += 1

cap.release()

#座標を生成
os.makedirs(output_dir, exist_ok=True)
command = f"{openpose_path} --image_dir {image_dir} --write_json {output_dir} --display 0 --render_pose 0"

process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

output, error = process.communicate()

print(output)

# 音声ファイルを読み込む
r = sr.Recognizer()
with sr.AudioFile(audio_path) as source:
    audio = r.record(source)

# 文章ファイルに変換
try:
    text = r.recognize_google(audio, language='ja-JP')
    print("Google Speech Recognition thinks you said: " + text)

    os.makedirs(txt_dir, exist_ok=True)
    with open(txt_path, 'w', encoding='utf-8') as file:
        file.write(text)

except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")
