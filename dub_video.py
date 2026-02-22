# dub_video.py — Supernan AI Dubbing Pipeline
# Full modular pipeline for English → Hindi video dubbing

import os
import whisper
import librosa
import soundfile as sf
import numpy as np
import subprocess
from transformers import pipeline
from TTS.api import TTS
import torch

# ── CONFIG ──────────────────────────────────────────────
INPUT_VIDEO    = "supernan_full.mp4"
CLIP_START     = "00:00:15"
CLIP_END       = "00:00:30"
CLIP_VIDEO     = "clip.mp4"
CLIP_AUDIO     = "clip_audio.wav"
REFERENCE_WAV  = "reference_voice.wav"
TRANSCRIPT_EN  = "transcript_en.txt"
TRANSCRIPT_HI  = "transcript_hi.txt"
HINDI_RAW      = "hindi_audio_raw.wav"
HINDI_SYNCED   = "hindi_audio_synced.wav"
OUTPUT_VIDEO   = "output_final.mp4"
# ────────────────────────────────────────────────────────

def extract_clip():
    """Cut 15-second segment and extract audio"""
    print("📹 Extracting clip...")
    subprocess.run(["ffmpeg","-i",INPUT_VIDEO,"-ss",CLIP_START,"-to",CLIP_END,
                    "-c:v","libx264","-c:a","aac",CLIP_VIDEO,"-y"])
    subprocess.run(["ffmpeg","-i",CLIP_VIDEO,"-vn","-acodec","pcm_s16le",
                    "-ar","16000","-ac","1",CLIP_AUDIO,"-y"])
    subprocess.run(["ffmpeg","-i",CLIP_VIDEO,"-ss","00:00:02","-to","00:00:10",
                    "-vn","-acodec","pcm_s16le","-ar","22050","-ac","1",REFERENCE_WAV,"-y"])
    print("✅ Clip extracted")

def transcribe_audio():
    """Transcribe English audio using Whisper large-v3"""
    print("🎙️ Transcribing...")
    model = whisper.load_model("large-v3")
    result = model.transcribe(CLIP_AUDIO, language="en", word_timestamps=True)
    text = result["text"]
    with open(TRANSCRIPT_EN,"w") as f:
        f.write(text)
    print(f"✅ Transcript: {text}")
    return text

def translate_text():
    """Translate English to Hindi"""
    print("🌐 Translating to Hindi...")
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi", device=0)
    with open(TRANSCRIPT_EN) as f:
        en = f.read()
    hi = translator(en, max_length=500)[0]['translation_text']
    with open(TRANSCRIPT_HI,"w") as f:
        f.write(hi)
    print(f"✅ Hindi: {hi}")
    return hi

def clone_voice():
    """Generate Hindi audio with voice cloning"""
    print("🗣️ Cloning voice...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    with open(TRANSCRIPT_HI) as f:
        hi = f.read()
    tts.tts_to_file(text=hi, speaker_wav=REFERENCE_WAV, language="hi", file_path=HINDI_RAW)
    print("✅ Hindi audio generated")

def sync_audio_duration():
    """Time-stretch Hindi audio to match original clip duration"""
    print("⏱️ Syncing audio duration...")
    target = 15.0
    audio, sr = librosa.load(HINDI_RAW, sr=None)
    duration = len(audio) / sr
    rate = duration / target
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    sf.write(HINDI_SYNCED, stretched, sr)
    print(f"✅ Synced ({duration:.1f}s → {target}s)")

def lip_sync_video():
    """Run VideoReTalking lip sync"""
    print("👄 Running lip sync...")
    subprocess.run(["python","video-retalking/inference.py",
                    "--face", CLIP_VIDEO,
                    "--audio", HINDI_SYNCED,
                    "--outfile", OUTPUT_VIDEO])
    print("✅ Lip sync complete")

def quality_check():
    """Score output sharpness"""
    import cv2
    cap = cv2.VideoCapture(OUTPUT_VIDEO)
    scores = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    cap.release()
    avg = np.mean(scores)
    print(f"🔍 Sharpness: {avg:.1f} — {'✅ GOOD' if avg > 100 else '⚠️ CHECK OUTPUT'}")

def main():
    extract_clip()
    transcribe_audio()
    translate_text()
    clone_voice()
    sync_audio_duration()
    lip_sync_video()
    quality_check()
    print("\n🎉 Pipeline complete! Output:", OUTPUT_VIDEO)

if __name__ == "__main__":
    main()
