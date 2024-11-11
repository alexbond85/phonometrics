from phonometrics.transcription.words.whisper_local import LocalWhisperModel
import torchaudio

def test_local_whisper_model(sample_audio_data):
    # Initialize the LocalWhisperModel
    model = LocalWhisperModel(model_size="base")
    path_to_file = sample_audio_data["path"]
    # Transcribe the audio file
    transcription = model.transcribe_from_file(path_to_file)
    # Check that the transcription matches the expected transcription
    expected_transcription = sample_audio_data["transcription_words"]
    computed_transcription = transcription["transcription"].lstrip()
    assert expected_transcription == computed_transcription




def test_local_whisper_model_from_waveform(sample_audio_data):
    # Initialize the LocalWhisperModel with "medium" size
    model = LocalWhisperModel(model_size="medium")

    # Load waveform and sample rate from sample audio data
    path_to_file = sample_audio_data["path"]
    expected_transcription = sample_audio_data["transcription_words"]

    # Read audio file and load waveform
    waveform, sample_rate = torchaudio.load(path_to_file)

    # Transcribe the audio waveform
    transcription = model.transcribe_from_waveform(waveform, sample_rate)

    # Compare the computed transcription with the expected transcription
    computed_transcription = (
        transcription["transcription"].strip().replace(",", "")
    )
    assert computed_transcription == expected_transcription.strip().replace(
        ",", ""
    ), f"Expected: {expected_transcription}, but got: {computed_transcription}"
