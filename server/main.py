"""Chatterbox TTS FastAPI Server.

A REST API server for Chatterbox TTS that supports both English and Multilingual models.
Designed to run on a GPU-enabled EC2 instance and be called remotely by the voice bot.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000

Or with Docker:
    docker-compose up
"""

import base64
import io
import os
import tempfile
import uuid
import time
from collections import OrderedDict
from typing import Optional, List
from functools import lru_cache

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import Response
from pydantic import BaseModel, Field
from loguru import logger

# Configure logging
logger.add("chatterbox_server.log", rotation="100 MB", level="INFO")

# Import Chatterbox models
try:
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
except ImportError:
    # If running from server directory, try relative import
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from chatterbox.tts import ChatterboxTTS
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES


# ============================================================================
# Configuration
# ============================================================================

# Environment variables
DEVICE = os.environ.get("DEVICE", None)
MAX_SPEAKER_CACHE = int(os.environ.get("MAX_SPEAKER_CACHE", 100))
LOAD_ENGLISH_MODEL = os.environ.get("LOAD_ENGLISH_MODEL", "true").lower() == "true"
LOAD_MULTILINGUAL_MODEL = os.environ.get("LOAD_MULTILINGUAL_MODEL", "true").lower() == "true"

# Auto-detect device
if DEVICE is None:
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

logger.info(f"Using device: {DEVICE}")

# Patch torch.load for MPS devices
if DEVICE == "mps":
    map_location = torch.device(DEVICE)
    torch_load_original = torch.load
    def patched_torch_load(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = map_location
        return torch_load_original(*args, **kwargs)
    torch.load = patched_torch_load


# ============================================================================
# LRU Speaker Cache
# ============================================================================

class SpeakerCache:
    """LRU cache for speaker conditionals to avoid re-cloning on every request."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.metadata = {}  # Store speaker metadata (audio path, model type, etc.)
    
    def get(self, speaker_id: str):
        """Get cached conditionals, moving to end (most recently used)."""
        if speaker_id in self.cache:
            self.cache.move_to_end(speaker_id)
            return self.cache[speaker_id], self.metadata.get(speaker_id)
        return None, None
    
    def set(self, speaker_id: str, conditionals, metadata: dict = None):
        """Cache conditionals with LRU eviction."""
        if speaker_id in self.cache:
            self.cache.move_to_end(speaker_id)
        else:
            if len(self.cache) >= self.max_size:
                # Evict oldest (first) item
                oldest_id = next(iter(self.cache))
                del self.cache[oldest_id]
                self.metadata.pop(oldest_id, None)
                logger.info(f"Evicted speaker {oldest_id} from cache")
        
        self.cache[speaker_id] = conditionals
        if metadata:
            self.metadata[speaker_id] = metadata
        logger.info(f"Cached speaker {speaker_id}, cache size: {len(self.cache)}")
    
    def remove(self, speaker_id: str):
        """Remove a specific speaker from cache."""
        if speaker_id in self.cache:
            del self.cache[speaker_id]
            self.metadata.pop(speaker_id, None)
    
    def clear(self):
        """Clear all cached speakers."""
        self.cache.clear()
        self.metadata.clear()
    
    def list_speakers(self) -> List[dict]:
        """List all cached speakers with metadata."""
        return [
            {"speaker_id": sid, **self.metadata.get(sid, {})}
            for sid in self.cache.keys()
        ]


# Global speaker cache
speaker_cache = SpeakerCache(max_size=MAX_SPEAKER_CACHE)


# ============================================================================
# Model Loading
# ============================================================================

# Global model instances
english_model: Optional[ChatterboxTTS] = None
multilingual_model: Optional[ChatterboxMultilingualTTS] = None


def load_models():
    """Load Chatterbox models on startup."""
    global english_model, multilingual_model
    
    if LOAD_ENGLISH_MODEL:
        logger.info("Loading English ChatterboxTTS model...")
        start = time.time()
        english_model = ChatterboxTTS.from_pretrained(device=DEVICE)
        logger.info(f"English model loaded in {time.time() - start:.2f}s")
    
    if LOAD_MULTILINGUAL_MODEL:
        logger.info("Loading Multilingual ChatterboxTTS model...")
        start = time.time()
        multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
        logger.info(f"Multilingual model loaded in {time.time() - start:.2f}s")


# ============================================================================
# Pydantic Models
# ============================================================================

class TTSRequest(BaseModel):
    """Request body for TTS generation using cached speaker."""
    text: str = Field(..., description="Text to synthesize")
    speaker_id: str = Field(..., description="ID of cached speaker from /clone_speaker")
    language_id: Optional[str] = Field(None, description="Language code for multilingual model (e.g., 'en', 'hi', 'fr')")
    exaggeration: float = Field(0.5, ge=0.0, le=2.0, description="Emotion exaggeration (0.5 is neutral)")
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0, description="CFG/Pace weight")
    temperature: float = Field(0.8, ge=0.05, le=5.0, description="Sampling temperature")
    return_format: str = Field("base64", description="Response format: 'base64' or 'raw'")


class TTSWithAudioRequest(BaseModel):
    """Request body for TTS generation with inline audio prompt."""
    text: str = Field(..., description="Text to synthesize")
    language_id: Optional[str] = Field(None, description="Language code for multilingual model")
    exaggeration: float = Field(0.5, ge=0.0, le=2.0)
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0)
    temperature: float = Field(0.8, ge=0.05, le=5.0)
    return_format: str = Field("base64", description="Response format: 'base64' or 'raw'")


class CloneSpeakerResponse(BaseModel):
    """Response from /clone_speaker endpoint."""
    speaker_id: str
    model_type: str
    message: str


class TTSResponse(BaseModel):
    """Response from /tts endpoint."""
    audio: str = Field(..., description="Base64 encoded PCM audio (int16, 24kHz, mono)")
    sample_rate: int = 24000
    duration_ms: float
    generation_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    device: str
    english_model_loaded: bool
    multilingual_model_loaded: bool
    cached_speakers: int
    supported_languages: dict


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Chatterbox TTS Server",
    description="REST API for Chatterbox Text-to-Speech with speaker cloning",
    version="1.0.0",
    docs_url="/",
)


@app.on_event("startup")
async def startup_event():
    """Load models on server startup."""
    load_models()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with model status."""
    return HealthResponse(
        status="healthy",
        device=DEVICE,
        english_model_loaded=english_model is not None,
        multilingual_model_loaded=multilingual_model is not None,
        cached_speakers=len(speaker_cache.cache),
        supported_languages=SUPPORTED_LANGUAGES if multilingual_model else {},
    )


@app.get("/speakers")
async def list_speakers():
    """List all cached speakers."""
    return {"speakers": speaker_cache.list_speakers()}


@app.delete("/speakers/{speaker_id}")
async def delete_speaker(speaker_id: str):
    """Remove a speaker from cache."""
    speaker_cache.remove(speaker_id)
    return {"message": f"Speaker {speaker_id} removed from cache"}


@app.post("/clone_speaker", response_model=CloneSpeakerResponse)
async def clone_speaker(
    audio_file: UploadFile = File(..., description="Reference audio file (WAV, MP3, etc.)"),
    model_type: str = Form("english", description="Model type: 'english' or 'multilingual'"),
    exaggeration: float = Form(0.5, description="Emotion exaggeration for conditioning"),
    speaker_id: Optional[str] = Form(None, description="Custom speaker ID (auto-generated if not provided)"),
):
    """
    Upload an audio file to clone a speaker voice.
    
    Returns a speaker_id that can be used with /tts endpoint for subsequent requests.
    The speaker conditionals are cached in memory for fast TTS generation.
    """
    # Validate model type
    if model_type == "english" and english_model is None:
        raise HTTPException(status_code=503, detail="English model not loaded")
    if model_type == "multilingual" and multilingual_model is None:
        raise HTTPException(status_code=503, detail="Multilingual model not loaded")
    
    # Generate speaker ID if not provided
    if speaker_id is None:
        speaker_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as tmp:
        content = await audio_file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Select model and prepare conditionals
        model = english_model if model_type == "english" else multilingual_model
        
        logger.info(f"Cloning speaker from {audio_file.filename}, model={model_type}")
        start = time.time()
        
        # Prepare conditionals (this does the heavy lifting)
        model.prepare_conditionals(tmp_path, exaggeration=exaggeration)
        
        # Cache the conditionals
        # We need to deep copy the conditionals since the model reuses the same object
        cached_conds = _copy_conditionals(model.conds, model_type)
        
        speaker_cache.set(speaker_id, cached_conds, {
            "model_type": model_type,
            "original_filename": audio_file.filename,
            "exaggeration": exaggeration,
            "created_at": time.time(),
        })
        
        clone_time = (time.time() - start) * 1000
        logger.info(f"Speaker cloned in {clone_time:.2f}ms, speaker_id={speaker_id}")
        
        return CloneSpeakerResponse(
            speaker_id=speaker_id,
            model_type=model_type,
            message=f"Speaker cloned successfully in {clone_time:.2f}ms",
        )
    
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


def _copy_conditionals(conds, model_type: str):
    """Deep copy conditionals to avoid reference issues."""
    # Import the Conditionals class based on model type
    if model_type == "english":
        from chatterbox.tts import Conditionals
    else:
        from chatterbox.mtl_tts import Conditionals
    
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    
    # Copy T3 conditionals
    t3_copy = T3Cond(
        speaker_emb=conds.t3.speaker_emb.clone() if conds.t3.speaker_emb is not None else None,
        cond_prompt_speech_tokens=conds.t3.cond_prompt_speech_tokens.clone() if conds.t3.cond_prompt_speech_tokens is not None else None,
        emotion_adv=conds.t3.emotion_adv.clone() if conds.t3.emotion_adv is not None else None,
        clap_emb=conds.t3.clap_emb.clone() if hasattr(conds.t3, 'clap_emb') and conds.t3.clap_emb is not None else None,
        cond_prompt_speech_emb=conds.t3.cond_prompt_speech_emb.clone() if hasattr(conds.t3, 'cond_prompt_speech_emb') and conds.t3.cond_prompt_speech_emb is not None else None,
    )
    
    # Copy gen dict
    gen_copy = {}
    for k, v in conds.gen.items():
        if torch.is_tensor(v):
            gen_copy[k] = v.clone()
        else:
            gen_copy[k] = v
    
    return Conditionals(t3_copy, gen_copy)


@app.post("/tts")
async def generate_tts(request: TTSRequest):
    """
    Generate speech from text using a cached speaker.
    
    Requires a speaker_id from a previous /clone_speaker call.
    """
    # Get cached conditionals
    cached_conds, metadata = speaker_cache.get(request.speaker_id)
    if cached_conds is None:
        raise HTTPException(
            status_code=404,
            detail=f"Speaker {request.speaker_id} not found. Call /clone_speaker first."
        )
    
    model_type = metadata.get("model_type", "english")
    
    # Select model
    if model_type == "english":
        if english_model is None:
            raise HTTPException(status_code=503, detail="English model not loaded")
        model = english_model
    else:
        if multilingual_model is None:
            raise HTTPException(status_code=503, detail="Multilingual model not loaded")
        model = multilingual_model
    
    # Validate language for multilingual
    if model_type == "multilingual" and request.language_id:
        if request.language_id.lower() not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: {request.language_id}. Supported: {list(SUPPORTED_LANGUAGES.keys())}"
            )
    
    logger.info(f"Generating TTS: text={request.text[:50]}..., speaker={request.speaker_id}")
    start = time.time()
    
    # Set cached conditionals on model
    model.conds = cached_conds
    
    # Update exaggeration if different from cached
    if request.exaggeration != cached_conds.t3.emotion_adv[0, 0, 0].item():
        from chatterbox.models.t3.modules.cond_enc import T3Cond
        model.conds.t3 = T3Cond(
            speaker_emb=cached_conds.t3.speaker_emb,
            cond_prompt_speech_tokens=cached_conds.t3.cond_prompt_speech_tokens,
            emotion_adv=request.exaggeration * torch.ones(1, 1, 1, dtype=cached_conds.t3.speaker_emb.dtype),
        ).to(device=DEVICE)
    
    # Generate audio
    if model_type == "english":
        wav = model.generate(
            request.text,
            audio_prompt_path=None,  # Use cached conditionals
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            temperature=request.temperature,
        )
    else:
        wav = model.generate(
            request.text,
            language_id=request.language_id or "en",
            audio_prompt_path=None,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            temperature=request.temperature,
        )
    
    generation_time = (time.time() - start) * 1000
    
    # Convert to numpy if tensor
    if isinstance(wav, torch.Tensor):
        wav = wav.squeeze(0).detach().cpu().numpy()
    
    # Convert to int16 PCM
    if wav.dtype != np.int16:
        wav = np.clip(wav, -1.0, 1.0)
        wav = (wav * 32767).astype(np.int16)
    
    # Calculate duration
    duration_ms = len(wav) / 24000 * 1000
    
    logger.info(f"Generated {duration_ms:.0f}ms audio in {generation_time:.0f}ms")
    
    # Return response
    if request.return_format == "raw":
        return Response(
            content=wav.tobytes(),
            media_type="audio/pcm",
            headers={
                "X-Sample-Rate": "24000",
                "X-Duration-Ms": str(duration_ms),
                "X-Generation-Time-Ms": str(generation_time),
            }
        )
    else:
        audio_b64 = base64.b64encode(wav.tobytes()).decode("utf-8")
        return TTSResponse(
            audio=audio_b64,
            sample_rate=24000,
            duration_ms=duration_ms,
            generation_time_ms=generation_time,
        )


@app.post("/tts_with_audio")
async def generate_tts_with_audio(
    text: str = Form(..., description="Text to synthesize"),
    audio_file: UploadFile = File(..., description="Reference audio file"),
    model_type: str = Form("english", description="Model type: 'english' or 'multilingual'"),
    language_id: Optional[str] = Form(None, description="Language code for multilingual model"),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5),
    temperature: float = Form(0.8),
    return_format: str = Form("base64"),
):
    """
    Generate speech with an inline audio prompt (no caching).
    
    Use this for one-off requests where you don't need to reuse the speaker.
    For repeated requests with the same speaker, use /clone_speaker + /tts instead.
    """
    # Validate model
    if model_type == "english" and english_model is None:
        raise HTTPException(status_code=503, detail="English model not loaded")
    if model_type == "multilingual" and multilingual_model is None:
        raise HTTPException(status_code=503, detail="Multilingual model not loaded")
    
    model = english_model if model_type == "english" else multilingual_model
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as tmp:
        content = await audio_file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        logger.info(f"Generating TTS with audio: text={text[:50]}...")
        start = time.time()
        
        # Generate with audio prompt
        if model_type == "english":
            wav = model.generate(
                text,
                audio_prompt_path=tmp_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )
        else:
            wav = model.generate(
                text,
                language_id=language_id or "en",
                audio_prompt_path=tmp_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )
        
        generation_time = (time.time() - start) * 1000
        
        # Convert to numpy if tensor
        if isinstance(wav, torch.Tensor):
            wav = wav.squeeze(0).detach().cpu().numpy()
        
        # Convert to int16 PCM
        if wav.dtype != np.int16:
            wav = np.clip(wav, -1.0, 1.0)
            wav = (wav * 32767).astype(np.int16)
        
        duration_ms = len(wav) / 24000 * 1000
        
        logger.info(f"Generated {duration_ms:.0f}ms audio in {generation_time:.0f}ms")
        
        if return_format == "raw":
            return Response(
                content=wav.tobytes(),
                media_type="audio/pcm",
                headers={
                    "X-Sample-Rate": "24000",
                    "X-Duration-Ms": str(duration_ms),
                    "X-Generation-Time-Ms": str(generation_time),
                }
            )
        else:
            audio_b64 = base64.b64encode(wav.tobytes()).decode("utf-8")
            return TTSResponse(
                audio=audio_b64,
                sample_rate=24000,
                duration_ms=duration_ms,
                generation_time_ms=generation_time,
            )
    
    finally:
        os.unlink(tmp_path)


@app.get("/languages")
async def get_languages():
    """Get list of supported languages for multilingual model."""
    if multilingual_model is None:
        raise HTTPException(status_code=503, detail="Multilingual model not loaded")
    return {"languages": SUPPORTED_LANGUAGES}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

