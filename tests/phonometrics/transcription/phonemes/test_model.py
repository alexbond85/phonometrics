from transformers import AutoModelForCTC
from transformers import AutoProcessor

from phonometrics.transcription.phonemes.model import TranscriptionModel


def test_transcription_model(sample_audio_data):
    """Tests the AudioTranscriber class for correct transcription and output
    validation.
    """
    # Initialize paths and model
    path_to_audio = sample_audio_data["path"]
    # Model setup
    model_name = "Cnam-LMSSC/wav2vec2-french-phonemizer"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCTC.from_pretrained(model_name)
    audio_transcriber = TranscriptionModel(model, processor)

    # Transcription process
    result = audio_transcriber.transcribe_from_file(path_to_audio)

    # Assertions
    expected_transcription = sample_audio_data["transcription_phonemes"]
    assert result["transcription"] == expected_transcription, (
        f"Expected transcription '{expected_transcription}', but got "
        f"'{result['transcription']}'"
    )

    # Further checks
    len_check = len(result["start_timestamps"])
    assert len_check == len(result["end_timestamps"]) == len(
        result["probabilities"]
    ), "Lengths of timestamps and probabilities do not match"
