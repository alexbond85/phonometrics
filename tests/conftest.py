import pytest
from pathlib import Path


@pytest.fixture
def sample_audio_data():
    current_directory = Path(__file__).parent
    path_to_input_file = current_directory / 'data' / 'billet.mp3'
    transcription = 'ɛl vøt alɛt o kɔ̃sɛʁ mɛz ɛl na pa də bijɛ'
    return {"path": path_to_input_file, "transcription": transcription}