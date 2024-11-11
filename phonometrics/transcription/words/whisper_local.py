from typing import Dict

import torch
import whisper  # type: ignore

from phonometrics.transcription.words.model import WordsTranscriptionModel


class LocalWhisperModel(WordsTranscriptionModel):
    """
    Local Whisper model implementation for transcribing audio files.

    Attributes
    ----------
    model : whisper.Model
        Pre-trained Whisper model for transcription.

    Methods
    -------
    transcribe_from_file(file_path: str) -> Dict[str, str]
        Transcribes an audio file using the locally loaded Whisper model.
    """

    def __init__(self, model_size: str = "base"):
        """
        Initializes the LocalWhisperModel with a specified model size.

        Parameters
        ----------
        model_size : str, optional
            Size of the Whisper model to load (default is "base").

        Notes
        -----
        If a GPU is available, the model will be loaded onto the CUDA device;
        otherwise, it defaults to CPU.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_size, device=device)

    def transcribe_from_file(self, file_path: str) -> Dict[str, str]:
        """
        Transcribes an audio file using the locally loaded Whisper model.

        Parameters
        ----------
        file_path : str
            Path to the audio file.

        Returns
        -------
        Dict[str, str]
            A dictionary containing the transcription text under the single key
            "transcription".

        Notes
        -----
        The audio waveform is loaded with `torchaudio`, converted to a
        NumPy array, and transcribed by the Whisper model.
        """
        result = self.model.transcribe(file_path)
        transcription = result.get("text", "")
        return {"transcription": transcription}
