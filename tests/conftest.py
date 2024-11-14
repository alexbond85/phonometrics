from pathlib import Path

import pytest
import yaml


@pytest.fixture
def sample_audio_data():
    current_directory = Path(__file__).parent
    path_to_input_file = current_directory / "data" / "billet.mp3"
    transcription_phonemes = "ɛl vøt alɛt o kɔ̃sɛʁ mɛz ɛl na pa də bijɛ"
    transcription_words = (
        "Elle veut aller au concert, mais elle n'a pas de billet."
    )
    return {
        "path": str(path_to_input_file),
        "transcription_phonemes": transcription_phonemes,
        "transcription_words": transcription_words,
    }


@pytest.fixture
def path_to_uncut_audio():
    current_directory = Path(__file__).parent
    path_to_input_file = current_directory / "data" / ".."/".." /"02-Chapitre_2.mp3"
    return path_to_input_file


@pytest.fixture
def tests_directory():
    return Path(__file__).parent


@pytest.fixture
def transcription_service_url(tests_directory):
    path_to_config = tests_directory / "../config.yaml"
    with open(path_to_config, "r") as f:
        config = yaml.safe_load(f)
    return config["transcription_service_url"]


@pytest.fixture
def word_transcription_service_url(tests_directory):
    path_to_config = tests_directory / "../config.yaml"
    with open(path_to_config, "r") as f:
        config = yaml.safe_load(f)
    return config["word_transcription_service_url"]


@pytest.fixture
def openai_word_transcription_service_url(tests_directory):
    path_to_config = tests_directory / "../config.yaml"
    with open(path_to_config, "r") as f:
        config = yaml.safe_load(f)
    return config["openai_word_transcription_service_url"]
