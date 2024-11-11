from phonometrics.transcription.words.whisper_openai import OpenAIWhisperModel


def test_openai_whisper_model(sample_audio_data):
    # Initialize the LocalWhisperModel
    model = OpenAIWhisperModel()
    path_to_file = sample_audio_data["path"]
    # Transcribe the audio file
    transcription = model.transcribe_from_file(path_to_file)
    # Check that the transcription matches the expected transcription
    expected_transcription = sample_audio_data["transcription_words"]
    computed_transcription = transcription["transcription"].lstrip()
    assert expected_transcription == computed_transcription
