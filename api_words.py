import io
import logging

from functools import lru_cache

import torchaudio  # type: ignore
import whisper  # type: ignore

from fastapi import FastAPI
from fastapi import File
from fastapi import Query
from fastapi import UploadFile
from fastapi.responses import JSONResponse

from phonometrics.transcription.words.whisper_local import LocalWhisperModel


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Initialize FastAPI
app = FastAPI()


# Initialize FastAPI
app = FastAPI()


# Cached model loader to ensure each model size is only loaded once
@lru_cache(maxsize=2)
def get_model(model_size: str) -> LocalWhisperModel:
    return LocalWhisperModel(model_size=model_size)


@app.post("/transcribe")
async def transcribe_audio(
    model_size: str = Query("base", enum=["base", "medium"]),
    file: UploadFile = File(...),
):
    # Load
    transcriber = get_model(model_size)
    logger.info("Received transcription request.")

    # Read the uploaded file into bytes
    audio_bytes = await file.read()
    logger.info("Audio file read into bytes.")

    # Load audio data using torchaudio
    audio_file = io.BytesIO(audio_bytes)
    audio_waveform, sample_rate = torchaudio.load(audio_file)
    logger.info(
        f"Audio data loaded: sample rate = {sample_rate}, waveform shape = {audio_waveform.shape}"
    )

    # Transcribe audio
    transcription = transcriber.transcribe_from_waveform(
        audio_waveform, sample_rate
    )
    logger.info("Transcription completed.")

    # Return the transcription as JSON
    return JSONResponse(content=transcription)
