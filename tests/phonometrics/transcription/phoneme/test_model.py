import os
from transformers import AutoModelForCTC, AutoProcessor
from phonometrics.transcription.phoneme.model import TranscriptionModel


def test_transcription_model():
    """Tests the AudioTranscriber class for correct transcription and output
    validation.
    """
    # Initialize paths and model
    current_folder = os.path.dirname(os.path.abspath(__file__))
    root_folder = os.path.abspath(
        os.path.join(current_folder, "../../..")
    )
    path_to_audio = os.path.join(root_folder, "data/billet.mp3")
    # Model setup
    model_name = "Cnam-LMSSC/wav2vec2-french-phonemizer"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCTC.from_pretrained(model_name)
    audio_transcriber = TranscriptionModel(model, processor)

    # Transcription process
    result = audio_transcriber.transcribe_from_file(path_to_audio)

    # Assertions
    expected_transcription = 'ɛl vøt alɛt o kɔ̃sɛʁ mɛz ɛl na pa də bijɛ'
    assert result["transcription"] == expected_transcription, (
        f"Expected transcription '{expected_transcription}', but got "
        f"'{result['transcription']}'"
    )

    # Further checks
    len_check = len(result["start_timestamps"])
    assert len_check == len(result["end_timestamps"]) == len(
        result["probabilities"]
    ), "Lengths of timestamps and probabilities do not match"
