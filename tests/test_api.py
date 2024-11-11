import pytest
import requests



@pytest.mark.integration
def test_phoneme_api(sample_audio_data, api_url):
    input_file = sample_audio_data["path"]
    files = {"file": ("billet.mp3", open(str(input_file), "rb"), "audio/wav")}
    response = requests.post(api_url + "transcribe/phonemes", files=files)
    assert response.status_code == 200, f"Request failed with status code {response.status_code}"
    response_data = response.json()
    assert sample_audio_data["transcription_phonemes"] == response_data["transcription"]

@pytest.mark.integration
def test_whisper_api(sample_audio_data, api_url):
    input_file = sample_audio_data["path"]
    files = {"file": ("billet.mp3", open(str(input_file), "rb"), "audio/wav")}
    params = {"model_size": "base"}
    response = requests.post(api_url + "transcribe/words", files=files, params=params)
    assert response.status_code == 200, f"Request failed with status code {response.status_code}"
    response_data = response.json()
    assert sample_audio_data["transcription_words"] == response_data["transcription"].strip()