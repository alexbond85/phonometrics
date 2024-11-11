# Use the official Python image with slim tag for a smaller footprint
FROM python:3.11-slim

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    cmake \
    libasound2-dev \
    libsndfile1-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8

# Set the working directory
WORKDIR /app

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy the pyproject.toml and poetry.lock files
COPY pyproject.toml poetry.lock* ./

# Install dependencies using Poetry
RUN poetry config virtualenvs.create false && poetry install --no-root --no-interaction --no-ansi

# Download the model during the build
RUN python -c "\
from transformers import AutoProcessor, AutoModelForCTC; \
AutoProcessor.from_pretrained('Cnam-LMSSC/wav2vec2-french-phonemizer'); \
AutoModelForCTC.from_pretrained('Cnam-LMSSC/wav2vec2-french-phonemizer'); \
"

# Copy the rest of the application code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "phoneme_transcription_api:app", "--host", "0.0.0.0", "--port", "8000"]
