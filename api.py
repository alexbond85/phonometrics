import io
import logging
from functools import lru_cache

import torchaudio  # type: ignore
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from transformers import AutoModelForCTC, AutoProcessor  # type: ignore

from phonometrics.transcription.phonemes.model import TranscriptionModel
from phonometrics.transcription.words.whisper_local import LocalWhisperModel
from phonometrics.transcription.words.whisper_openai import OpenAIWhisperModel

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Transcriber setup
MODEL_NAME = "Cnam-LMSSC/wav2vec2-french-phonemizer"
logger.info("Initializing transcriber...")
transcriber = TranscriptionModel(
    model=AutoModelForCTC.from_pretrained(MODEL_NAME),
    processor=AutoProcessor.from_pretrained(MODEL_NAME)
)
logger.info("Transcriber ready.")


@app.post("/transcribe/phonemes")
async def transcribe_phonemes(file: UploadFile = File(...)):
    logger.info("Processing phoneme transcription request")
    audio_waveform, sample_rate = await extract_audio(file)
    transcription = transcriber.transcribe_from_waveform(audio_waveform, sample_rate)
    logger.info("Phoneme transcription completed")
    logger.info(f"Transcription: {transcription}")
    return JSONResponse(content=transcription)


@app.post("/transcribe/words")
async def transcribe_words(
    model_size: str = Query("base", enum=["base", "medium"]),
    file: UploadFile = File(...),
):
    whisper_model = load_whisper_model(model_size)
    logger.info("Processing word transcription request")
    audio_waveform, sample_rate = await extract_audio(file)
    transcription = whisper_model.transcribe_from_waveform(audio_waveform, sample_rate)
    logger.info("Word transcription completed")
    logger.info(f"Transcription: {transcription}")
    return JSONResponse(content=transcription)


@app.post("/transcribe/words/openai")
async def transcribe_words(
    file: UploadFile = File(...),
):
    whisper_model = load_openai_whisper_model()
    logger.info("Processing word transcription request")
    file_content = io.BytesIO(await file.read())
    file_content.name = file.filename
    transcription = whisper_model.transcribe_from_binary(file_content)
    logger.info("Word transcription completed")
    logger.info(f"Transcription: {transcription}")
    return JSONResponse(content=transcription)


@app.get("/health")
async def health_check():
    logger.info("Health check accessed")
    return {"status": "ok"}


@lru_cache(maxsize=2)
def load_whisper_model(model_size: str) -> LocalWhisperModel:
    """Loads the Whisper model based on specified size."""
    return LocalWhisperModel(model_size=model_size)


@lru_cache(maxsize=1)
def load_openai_whisper_model() -> OpenAIWhisperModel:
    """Loads the Whisper model based on specified size."""
    from dotenv import load_dotenv
    import os
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    logger.info("Loaded OpenAI API key")
    return OpenAIWhisperModel(api_key=api_key)


async def extract_audio(file: UploadFile):
    """Converts uploaded file into waveform and sample rate."""
    audio_bytes = await file.read()
    audio_waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
    logger.info(f"Audio loaded: sample rate = {sample_rate}, waveform shape = {audio_waveform.shape}")
    return audio_waveform, sample_rate
