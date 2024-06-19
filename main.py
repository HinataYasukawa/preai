import subprocess
import cv2
import speech_recognition as sr
import os
import json
from moviepy.editor import VideoFileClip
from pydub import AudioSegment

#動画ファイルを音声ファイルに
def extract_audio_from_video(video_path, audio_path):
    if not os.path.exists(audio_path):
        video = VideoFileClip(video_path)
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        video.audio.write_audiofile(audio_path)

#動画ファイルを画像ファイルに
def extract_frames_from_video(video_path, image_dir, fps):
    cap = cv2.VideoCapture(video_path)
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

#画像にopenposeをかけてjsonファイルに
def generate_pose_coordinates(openpose_path, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    command = f"{openpose_path} --image_dir {image_dir} --write_json {output_dir} --display 0 --render_pose 0"
    process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)

#audioファイルを分割
def split_audio(audio_path, output_dir, base_name, duration_ms=1 * 60 * 1000):
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

#auidioファイルを文字起こし
def transcribe_audio_chunks(audio_chunks, language='ja-JP'):
    r = sr.Recognizer()
    full_text = ""

    for i, chunk_path in enumerate(audio_chunks):
        with sr.AudioFile(chunk_path) as source:
            audio = r.record(source)
        try:
            text = r.recognize_google(audio, language=language)
            full_text += text + " "
            print(f"Chunk {i+1}: " + text)
        except sr.UnknownValueError:
            print(f"Google Speech Recognition could not understand audio for chunk {i+1}")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service for chunk {i+1}; {e}")

    return full_text

#audioファイルを結合
def combine_audio_chunks(audio_chunks, output_path):
    combined_audio = AudioSegment.empty()
    for chunk_path in audio_chunks:
        combined_audio += AudioSegment.from_wav(chunk_path)
    combined_audio.export(output_path, format="wav")

#テキストをファイル形式にして保存
def save_transcription(txt_path, text):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, 'w', encoding='utf-8') as file:
        file.write(text)

def delete_audio_files(audio_chunks):
    for chunk_path in audio_chunks:
        print(chunk_path)
        os.remove(chunk_path)

def main():
    number = "33"
    openpose_path = "bin\OpenPoseDemo.exe"
    video_path = "C:/openpose/examples/" + number + ".mp4"

    output_dir = "output/json"
    image_dir = "output/image"
    audio_dir = "output/audio"
    txt_dir = "output/txt"

    audio_path = os.path.join(audio_dir, number + ".wav")
    txt_path = os.path.join(txt_dir, number + ".txt")

    extract_audio_from_video(video_path, audio_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    extract_frames_from_video(video_path, image_dir, fps)
    generate_pose_coordinates(openpose_path, image_dir, output_dir)

    audio_chunks = split_audio(audio_path, audio_dir, number)
    combine_audio_chunks(audio_chunks, audio_path)

    if not os.path.exists(txt_path):
        full_text = transcribe_audio_chunks(audio_chunks)
        save_transcription(txt_path, full_text)

    delete_audio_files(audio_chunks)
    print(f"Final audio and text files saved as {audio_path} and {txt_path}")

if __name__ == "__main__":
    main()
