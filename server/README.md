# Chatterbox TTS Server

A FastAPI-based REST API server for Chatterbox Text-to-Speech with speaker cloning support.

## Features

- **English Model**: High-quality English TTS
- **Multilingual Model**: Support for 23 languages
- **Speaker Cloning**: Clone any voice from a reference audio file
- **Speaker Caching**: LRU cache for fast repeated TTS with same speaker
- **GPU Acceleration**: CUDA support for fast inference
- **Docker Deployment**: Production-ready Docker images

## Quick Start

### Local Development

```bash
cd chatterbox/server

# Install dependencies
pip install -r requirements.txt

# Install Chatterbox from parent directory
pip install -e ../

# Run the server
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker Deployment (GPU)

```bash
cd chatterbox/server

# Build and run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f
```

### Docker Deployment (CPU)

```bash
# Build CPU image
docker build -f Dockerfile.cpu -t chatterbox-tts-cpu ..

# Run
docker run -p 8000:8000 -v chatterbox_cache:/app/.cache/huggingface chatterbox-tts-cpu
```

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "device": "cuda",
  "english_model_loaded": true,
  "multilingual_model_loaded": true,
  "cached_speakers": 2,
  "supported_languages": {"en": "English", "hi": "Hindi", ...}
}
```

### Clone Speaker

Upload a reference audio file to clone a speaker voice:

```bash
curl -X POST http://localhost:8000/clone_speaker \
  -F "audio_file=@reference.wav" \
  -F "model_type=english" \
  -F "exaggeration=0.5"
```

Response:
```json
{
  "speaker_id": "a1b2c3d4",
  "model_type": "english",
  "message": "Speaker cloned successfully in 1234.56ms"
}
```

### Generate TTS

Generate speech using a cached speaker:

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a test.",
    "speaker_id": "a1b2c3d4",
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "return_format": "base64"
  }'
```

Response:
```json
{
  "audio": "base64_encoded_pcm_audio...",
  "sample_rate": 24000,
  "duration_ms": 1500.0,
  "generation_time_ms": 456.78
}
```

For raw PCM response:
```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "speaker_id": "a1b2c3d4", "return_format": "raw"}' \
  -o output.pcm
```

### Generate TTS with Audio (One-shot)

For one-off requests without caching:

```bash
curl -X POST http://localhost:8000/tts_with_audio \
  -F "text=Hello world" \
  -F "audio_file=@reference.wav" \
  -F "model_type=english" \
  -F "return_format=base64"
```

### List/Delete Speakers

```bash
# List all cached speakers
curl http://localhost:8000/speakers

# Delete a speaker
curl -X DELETE http://localhost:8000/speakers/a1b2c3d4
```

### Get Supported Languages

```bash
curl http://localhost:8000/languages
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | auto | Device to use: `cuda`, `mps`, or `cpu` |
| `LOAD_ENGLISH_MODEL` | `true` | Load English ChatterboxTTS |
| `LOAD_MULTILINGUAL_MODEL` | `true` | Load Multilingual model |
| `MAX_SPEAKER_CACHE` | `100` | Max cached speakers (LRU eviction) |
| `HF_TOKEN` | - | HuggingFace token (optional) |

## Audio Format

- **Sample Rate**: 24000 Hz
- **Channels**: Mono (1 channel)
- **Format**: 16-bit signed integer PCM

To convert to WAV:
```bash
# Using ffmpeg
ffmpeg -f s16le -ar 24000 -ac 1 -i output.pcm output.wav

# Using Python
import numpy as np
import scipy.io.wavfile as wav

pcm_data = np.frombuffer(open('output.pcm', 'rb').read(), dtype=np.int16)
wav.write('output.wav', 24000, pcm_data)
```

## Multilingual Support

Supported languages: Arabic, Danish, German, Greek, English, Spanish, Finnish, French, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Dutch, Norwegian, Polish, Portuguese, Russian, Swedish, Swahili, Turkish, Chinese.

Use the `language_id` parameter with the language code (e.g., `"hi"` for Hindi).

## Performance Tips

1. **Pre-clone speakers**: Use `/clone_speaker` to cache speaker conditionals
2. **Batch processing**: The speaker cache avoids re-cloning on each request
3. **GPU memory**: Each model uses ~4-6GB VRAM
4. **CPU mode**: Expect 5-10x slower inference on CPU

