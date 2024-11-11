from pathlib import Path

import pytest
import requests
import yaml


@pytest.mark.integration
def test_phoneme_api(sample_audio_data, tests_directory):
    # Define the URL for the API endpoint
    path_to_config = tests_directory / "../config.yaml"
    with open(path_to_config, "r") as f:
        config = yaml.safe_load(f)
    url = config["transcription_service_url"]
    # Determine the current directory and path to the input file
    current_directory = Path(__file__).parent
    path_to_input_file = current_directory / "data" / "billet.mp3"
    # Check if the file exists
    assert (
        path_to_input_file.exists()
    ), f"Input file {path_to_input_file} does not exist."
    # Prepare the file data for the request
    files = {
        "file": (
            "billet.mp3",
            open(str(path_to_input_file), "rb"),
            "audio/wav",
        )
    }
    # Send the POST request
    response = requests.post(url, files=files)
    # Check that the response was successful
    assert (
        response.status_code == 200
    ), f"Request failed with status code {response.status_code}"
    response_data = response.json()
    # assert
    assert sample_audio_data["transcription"] == response_data["transcription"]


@pytest.mark.integration
def test_whisper_api(sample_audio_data, tests_directory):
    # Define the URL for the API endpoint
    path_to_config = tests_directory / "../config.yaml"
    with open(path_to_config, "r") as f:
        config = yaml.safe_load(f)
    url = config["transcription_service_url"]
    # Determine the current directory and path to the input file
    current_directory = Path(__file__).parent
    path_to_input_file = current_directory / "data" / "billet.mp3"
    # Check if the file exists
    assert (
        path_to_input_file.exists()
    ), f"Input file {path_to_input_file} does not exist."
    # Prepare the file data for the request
    files = {
        "file": (
            "billet.mp3",
            open(str(path_to_input_file), "rb"),
            "audio/wav",
        )
    }
    params = {
        "model_size": "base"  #
    }
    # Send the POST request
    response = requests.post(url, files=files, params=params)
    # Check that the response was successful
    assert (
        response.status_code == 200
    ), f"Request failed with status code {response.status_code}"
    response_data = response.json()
    # assert
    assert sample_audio_data["transcription_words"] == response_data["transcription"]
