import subprocess
import cv2
import speech_recognition as sr
import os
from moviepy.editor import VideoFileClip

audio_path = "C:/openpose/examples/audio/audio.mp3"
video_path = "C:/openpose/examples/video/video.avi"
save_dir = "C:/openpose/examples/image"
openpose_path = "bin\OpenPoseDemo.exe"
image_dir = "examples\image"
output_dir = "output"
voice_path = "C:/openpose/examples/audio/sample.wav"

video = VideoFileClip(video_path)
video.audio.write_audiofile(audio_path)

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


command = f"{openpose_path} --image_dir {image_dir} --write_json {output_dir} --display 0 --render_pose 0"

process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

output, error = process.communicate()

print(output)


r = sr.Recognizer()
with sr.AudioFile(voice_path) as source:
    audio = r.record(source)

text = r.recognize_google(audio, language='ja-JP')
print(text)