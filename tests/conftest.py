from pathlib import Path

import pytest


@pytest.fixture
def sample_audio_data():
    current_directory = Path(__file__).parent
    path_to_input_file = current_directory / "data" / "billet.mp3"
    transcription = "ɛl vøt alɛt o kɔ̃sɛʁ mɛz ɛl na pa də bijɛ"
    transcription_words = (
        "Elle veut aller au concert, mais elle n'a pas de billet."
    )
    return {
        "path": str(path_to_input_file),
        "transcription": transcription,
        "transcription_words": transcription_words,
    }


@pytest.fixture
def tests_directory():
    return Path(__file__).parent
