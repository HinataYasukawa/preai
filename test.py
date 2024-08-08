import subprocess
import cv2
import speech_recognition as sr
import os
import json
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import whisper
import numpy as np
from whisper.audio import load_audio
import os
import whisper
from whisper.audio import HAS_FFMPEG, FFMPEG_BINARY

# ffmpegのパスを設定
FFMPEG_BINARY = r"C:\ffmpeg\bin\ffmpeg.exe"
if not HAS_FFMPEG:
    raise RuntimeError("ffmpeg not found. Install it or set the correct path in the FFMPEG_BINARY variable.")
model = whisper.load_model("small")

#動画ファイルを音声ファイルに
def extract_audio_from_video(video_path, audio_path):
    if not os.path.exists(audio_path):
        video = VideoFileClip(video_path)
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        video.audio.write_audiofile(audio_path)

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
def transcribe_audio_chunks(audio_chunks):
    audio_data = []
    for chunk_path in audio_chunks:
        audio = load_audio(chunk_path)
        audio_data.append(audio)
    full_audio = np.concatenate(audio_data, axis=0)
    result = model.transcribe(full_audio, language='ja')
    print(result["text"])
    return result

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
    number = "009"
    openpose_path = "bin\OpenPoseDemo.exe"
    video_path = "C:/openpose/test/" + number + ".mp4"

    output_dir = "test/json/" + number
    image_dir = "output/image"
    audio_dir = "test/audio"
    txt_dir = "test/txt"

    audio_path = os.path.join(audio_dir, number + ".wav")
    txt_path = os.path.join(txt_dir, number + ".txt")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    extract_audio_from_video(video_path, audio_path)
    audio_chunks = split_audio(audio_path, audio_dir, number)
    combine_audio_chunks(audio_chunks, audio_path)

    full_text = transcribe_audio_chunks(audio_chunks)
    save_transcription(txt_path, full_text)

    delete_audio_files(audio_chunks)
    print(f"Final audio and text files saved as {audio_path} and {txt_path}")
    print(f"processed {video_path}")

if __name__ == "__main__":
    main()
