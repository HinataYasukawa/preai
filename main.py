import subprocess
import cv2
import speech_recognition as sr
import os
import json
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

number = "32"
openpose_path = "bin\OpenPoseDemo.exe"
video_path = "C:/openpose/examples/" + number + ".mp4"

output_dir = "output/json"
image_dir = "output/image"
audio_dir = "output/audio"
txt_dir = "output/txt"

audio_path = os.path.join(audio_dir, number + ".wav")
txt_path = os.path.join(txt_dir, number + ".txt")

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

# 座標を生成
os.makedirs(output_dir, exist_ok=True)
command = f"{openpose_path} --image_dir {image_dir} --write_json {output_dir} --display 0 --render_pose 0"

process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

output, error = process.communicate()

print(output)

# 音声ファイルを1分毎に区切る
def split_audio(audio_path, output_dir, base_name, duration_ms=3 * 60 * 1000):
    audio = AudioSegment.from_wav(audio_path)
    audio_chunks = []
    for i in range(0, len(audio), duration_ms):
        chunk = audio[i:i + duration_ms]
        chunk_name = f"{base_name}_{i // duration_ms + 1}.wav"
        print(f"{chunk_name}")
        chunk_path = os.path.join(output_dir, chunk_name)
        chunk.export(chunk_path, format="wav")
        audio_chunks.append(chunk_path)
    return audio_chunks

# 音声ファイルを1分毎に区切り、文字起こしを行う
audio_chunks = split_audio(audio_path, audio_dir, number)
full_text = ""

r = sr.Recognizer()

for i, chunk_path in enumerate(audio_chunks):
    with sr.AudioFile(chunk_path) as source:
        audio = r.record(source)
        print("BBB")
    try:
        text = r.recognize_google(audio, language='ja-JP')
        print("AAA")
        full_text += text + " "
        print(f"Chunk {i+1}: " + text)
    except sr.UnknownValueError:
        print(f"Google Speech Recognition could not understand audio for chunk {i+1}")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service for chunk {i+1}; {e}")

# 区切られた音声ファイルを結合
combined_audio = AudioSegment.empty()
for chunk_path in audio_chunks:
    combined_audio += AudioSegment.from_wav(chunk_path)

combined_audio.export(audio_path, format="wav")

# 文字起こし結果を結合してファイルに保存
os.makedirs(txt_dir, exist_ok=True)
with open(txt_path, 'w', encoding='utf-8') as file:
    file.write(full_text)

print(f"Final audio and text files saved as {audio_path} and {txt_path}")
