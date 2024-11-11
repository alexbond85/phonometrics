from phonometrics.transcription.word.model import LocalWhisperModel


def test_local_whisper_model(sample_audio_data):

    model = LocalWhisperModel(model_size="base")
    path_to_file = sample_audio_data["path"]
    transcription = model.transcribe_from_file(path_to_file)
    expected_transcription = sample_audio_data["transcription_words"]
    computed_transcription = transcription["transcription"].lstrip()
    assert expected_transcription == computed_transcription
