import subprocess
import cv2
import speech_recognition as sr
import os
import json
from moviepy.editor import VideoFileClip

openpose_path = "bin\OpenPoseDemo.exe"
video_path = "C:/openpose/examples/video/01.mp4"
save_dir = "C:/openpose/examples/image"
output_dir = "output"

image_dir = "examples\image"
voice_path = "C:/openpose/examples/audio/sample.wav"
audio_path = "C:/openpose/examples/audio/audio.mp3"

video = VideoFileClip(video_path)

#video.audio.write_audiofile(audio_path)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

frame_count = 0
image_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    if frame_count % int(fps) == 0:
        image_path = os.path.join(save_dir, f"frame_{image_count:04d}.jpg")
        cv2.imwrite(image_path, frame)
        image_count += 1

    frame_count += 1

cap.release()

#jsonファイルに座標書き出し
command = f"{openpose_path} --image_dir {image_dir} --write_json {output_dir} --display 0 --render_pose 0"

process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

output, error = process.communicate()

print(output)

#動画を音声ファイルに変換
r = sr.Recognizer()
with sr.AudioFile(voice_path) as source:
    audio = r.record(source)

#音声ファイルの読み込み
r = sr.Recognizer()
with sr.AudioFile(audio_path) as source:
    audio = r.record(source)

#文章ファイルに変換
try:
    text = r.recognize_google(audio, language='ja-JP')
    print("Google Speech Recognition thinks you said: " + text)

    with open('output.txt', 'w', encoding='utf-8') as file:
        file.write(text)

except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")