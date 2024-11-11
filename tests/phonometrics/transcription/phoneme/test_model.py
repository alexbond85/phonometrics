import os
import torchaudio
from transformers import AutoProcessor, AutoModelForCTC
from phonometrics.transcription.phoneme.model import TranscriptionModel, Waveform
from phonometrics.transcription.phoneme.model import AudioTranscriber


def test_transcription_model():
    # init
    current_folder = os.path.dirname(os.path.abspath(__file__))
    # Go 3 folders up to get to the root folder
    root_folder = os.path.abspath(os.path.join(current_folder, "../../.."))
    path_to_audio = os.path.join(root_folder, "data/billet.mp3")
    model_name = "Cnam-LMSSC/wav2vec2-french-phonemizer"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCTC.from_pretrained(model_name)
    audio_transcriber = AudioTranscriber(TranscriptionModel(model, processor))
    # Load audio file and convert to waveform
    torchaudio.set_audio_backend("soundfile")
    audio_waveform, sample_rate = torchaudio.load(path_to_audio)
    print("Audio waveform loaded successfully.")
    #
    r = audio_transcriber.transcribe(Waveform(audio_waveform, sample_rate))
    assert r["transcription"] == 'ɛl vøt alɛt o kɔ̃sɛʁ mɛz ɛl na pa də bijɛ'
    l = len(r["start_timestamps"])
    assert l == len(r["end_timestamps"])
    assert l == len(r["probabilities"])