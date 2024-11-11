from abc import ABC
from abc import abstractmethod
from typing import Dict

import openai  # type: ignore
import torch
import whisper  # type: ignore


class WordsTranscriptionModel(ABC):
    """
    Abstract base class for transcription models that transcribe audio files.

    Methods
    -------
    transcribe_from_file(file_path: str) -> Dict[str, str]
        Transcribes an audio file and returns the transcription as a dictionary
        with a single key, "transcription".
    """

    @abstractmethod
    def transcribe_from_file(self, file_path: str) -> Dict[str, str]:
        """
        Abstract method to transcribe an audio file.

        Parameters
        ----------
        file_path : str
            Path to the audio file.

        Returns
        -------
        Dict[str, str]
            A dictionary containing the transcription with a single key,
            "transcription", where the value is the transcription text.
        """
        pass


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
        # Load the audio file
        # audio_waveform, sample_rate = torchaudio.load(file_path)
        # Convert to numpy array
        # audio_waveform = audio_waveform.squeeze().numpy()
        # Transcribe audio waveform
        result = self.model.transcribe(file_path)
        transcription = result.get("text", "")
        return {"transcription": transcription}


class OpenAIWhisperModel(WordsTranscriptionModel):
    """
    Whisper model implementation for transcribing audio files via OpenAI API.

    Attributes
    ----------
    client : openai.OpenAI
        OpenAI client for accessing Whisper API.

    Methods
    -------
    transcribe_from_file(file_path: str) -> Dict[str, str]
        Transcribes an audio file using OpenAI's Whisper API.
    """

    def __init__(self):
        """
        Initializes the OpenAIWhisperModel by creating an OpenAI API client.
        """
        self.client = openai.OpenAI()

    def transcribe_from_file(self, file_path: str) -> Dict[str, str]:
        """
        Transcribes an audio file using OpenAI's Whisper API.

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
        Opens the audio file in binary read mode and sends it
        to the OpenAI Whisper API.
        """
        with open(file_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )
            # transcription = transcription.get("text", "")
            return {"transcription": transcription}
