import pytest
import requests
from pathlib import Path


@pytest.mark.integration
def test_phoneme_api(sample_audio_data):
    # Define the URL for the API endpoint
    url = "http://localhost:8000/transcribe"
    # Determine the current directory and path to the input file
    current_directory = Path(__file__).parent
    path_to_input_file = current_directory / 'data' / 'billet.mp3'
    # Check if the file exists
    assert path_to_input_file.exists(), f"Input file {path_to_input_file} does not exist."
    # Prepare the file data for the request
    files = {
        'file': (
            'billet.mp3',
            open(str(path_to_input_file), 'rb'),
            'audio/wav'
        )
    }
    # Send the POST request
    response = requests.post(url, files=files)
    # Check that the response was successful
    assert response.status_code == 200, f"Request failed with status code {response.status_code}"
    response_data = response.json()
    # assert
    assert sample_audio_data["transcription"] == response_data["transcription"]
