import io
import logging

import torchaudio  # type: ignore
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoModelForCTC  # type: ignore
from transformers import AutoProcessor
from phonometrics.transcription.phoneme.model import TranscriptionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize the transcriber
model_name = "Cnam-LMSSC/wav2vec2-french-phonemizer"
logger.info("Loading processor and model...")
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForCTC.from_pretrained(model_name)
transcriber = TranscriptionModel(model=model, processor=processor)
logger.info("Processor and model loaded successfully.")


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    logger.info("Received transcription request.")

    # Read the uploaded file into bytes
    audio_bytes = await file.read()
    logger.info("Audio file read into bytes.")

    # Load audio data using torchaudio
    audio_file = io.BytesIO(audio_bytes)
    audio_waveform, sample_rate = torchaudio.load(audio_file)
    logger.info(f"Audio data loaded: sample rate = {sample_rate}, waveform shape = {audio_waveform.shape}")

    # Transcribe audio
    transcription = transcriber.transcribe_from_waveform(audio_waveform, sample_rate)
    logger.info("Transcription completed.")

    # Return the transcription as JSON
    return JSONResponse(content=transcription)


@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed.")
    return {"status": "ok"}
