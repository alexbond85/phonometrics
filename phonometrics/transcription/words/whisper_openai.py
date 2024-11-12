from typing import BinaryIO
from typing import Dict

import openai  # type: ignore

from phonometrics.transcription.words.model import WordsTranscriptionModel


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

    def __init__(self, api_key: str):
        """
        Initializes the OpenAIWhisperModel by creating an OpenAI API client.
        """
        self.client = openai.OpenAI(api_key=api_key)

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
        """
        with open(file_path, "rb") as audio_file:
            return self.transcribe_from_binary(audio_file)

    def transcribe_from_binary(self, audio_file: BinaryIO) -> Dict[str, str]:
        """
        Transcribes an audio file using OpenAI's Whisper API.

        Parameters
        ----------
        audio_file : BinaryIO
            Audio file bytes.

        Returns
        -------
        Dict[str, str]
            A dictionary containing the transcription text under the single key
            "transcription".
        """
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        ).text
        return {"transcription": transcription}
