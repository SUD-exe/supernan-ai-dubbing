# Supernan AI Dubbing Pipeline
Automated English → Hindi video dubbing with voice cloning and lip sync, built entirely with free/open-source tools.

## What This Does
Takes an English training video and produces a Hindi-dubbed version where:
- The voice speaks natural Hindi (cloned from the original speaker's voice)
- The lips in the video are synced to match the Hindi audio
- Face quality is restored after lip sync processing

## Pipeline Architecture
```
Input Video
    ↓
1. extract_clip()        → ffmpeg cuts 0:15–0:30 segment
    ↓
2. transcribe_audio()    → Whisper large-v3 converts speech to text
    ↓
3. translate_text()      → Helsinki-NLP opus-mt-en-hi translates to Hindi
    ↓
4. clone_voice()         → Coqui XTTS v2 generates Hindi audio in original speaker's voice
    ↓
5. sync_audio_duration() → librosa time_stretch matches Hindi audio to video length
    ↓
6. lip_sync_video()      → VideoReTalking syncs mouth movements to new audio
    ↓
7. quality_check()       → OpenCV measures sharpness of output frames
    ↓
Output Video (Hindi dubbed, lip synced)
```

## Tools Used (All Free)
| Task | Tool | Cost |
|------|------|------|
| Transcription | OpenAI Whisper large-v3 | Free |
| Translation | Helsinki-NLP opus-mt-en-hi | Free |
| Voice Cloning | Coqui XTTS v2 | Free |
| Audio Sync | librosa time_stretch | Free |
| Lip Sync | VideoReTalking | Free |
| Face Restoration | GFPGAN (built into VideoReTalking) | Free |
| Compute | Google Colab T4 GPU | Free |
| **Total Cost** | | **₹0** |

## Setup Instructions

### Requirements
- Google Colab (free tier) with T4 GPU enabled
- OR local machine with NVIDIA GPU + CUDA

### Install Dependencies
```bash
pip install openai-whisper
pip install transformers sentencepiece sacremoses
pip install TTS
pip install librosa soundfile
pip install pydub
apt-get install ffmpeg
git clone https://github.com/OpenTalker/video-retalking.git
cd video-retalking && pip install -r requirements.txt
```

### Run the Pipeline
```bash
python dub_video.py
```

Make sure your input video is named `supernan_full.mp4` and is in the same folder.

## Output
- Processed video: `output_final.mp4`
- English transcript: `transcript_en.txt`
- Hindi transcript: `transcript_hi.txt`

## Cost if Scaled (Per Minute of Video)
| Resource | Details | Cost |
|----------|---------|------|
| Vast.ai A100 GPU | ~8 mins processing per 1 min video | ~$0.04/min |
| Storage (S3) | Input + output video | ~$0.001/min |
| **Total** | | **~$0.04/min (~₹3.5/min)** |

**For 500 hours of video overnight:**
- Split into 5-minute chunks → 6,000 jobs
- Run 20 parallel A100 instances on Vast.ai
- Each instance processes ~300 chunks
- Estimated total time: 6–8 hours
- Estimated total cost: ~$1,200 (~₹1,00,000)

## Known Limitations
- XTTS v2 Hindi quality drops on sentences longer than 20 words
- VideoReTalking struggles with fast head movements or side profiles
- Audio time-stretch above 1.3x speed starts to sound unnatural
- Colab free tier disconnects after ~90 mins of inactivity
- Pipeline currently processes one segment at a time (not parallelized)

## What I'd Improve With More Time
- Replace Helsinki-NLP with IndicTrans2 for more natural Indian language translation
- Use WhisperX for more precise word-level timestamps
- Add automatic retry if sharpness score falls below threshold
- Add batching logic to handle videos longer than 30 minutes
- Add a simple web UI using Gradio so non-technical users can run it
- Dockerize the pipeline for one-command setup

## Scaling Architecture (500 Hours Overnight)
```
Video Files (S3)
      ↓
Job Queue (Redis + Celery)
      ↓
20x Vast.ai A100 Workers (parallel)
   Each worker runs full dub_video.py pipeline
      ↓
Output Files → S3
      ↓
Notification (email/Slack when done)
```

## Project Structure
```
supernan-ai-dubbing/
├── dub_video.py          # Main pipeline script
├── README.md             # This file
├── requirements.txt      # All dependencies
├── transcript_en.txt     # Generated English transcript
├── transcript_hi.txt     # Generated Hindi transcript
└── output_final.mp4      # Final dubbed video
```

## Author
Built for Supernan AI Automation Intern Challenge.
```

---

## Also Create a `requirements.txt` file

Same way — **Add file** → **Create new file** → name it `requirements.txt`:
```
openai-whisper
transformers
sentencepiece
sacremoses
TTS
librosa
soundfile
pydub
torch
torchvision
opencv-python
gdown
numpy
```

---

## Your GitHub Should Now Have These Files

Once you upload your code and output video later:
```
supernan-ai-dubbing/
├── README.md         ✅ done now
├── requirements.txt  ✅ done now
├── dub_video.py      ← add after pipeline runs
└── output_final.mp4  ← add after pipeline runs
