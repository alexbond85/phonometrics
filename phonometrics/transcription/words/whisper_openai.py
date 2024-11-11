import os

from typing import Dict

import openai  # type: ignore

from dotenv import load_dotenv

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

    def __init__(self):
        """
        Initializes the OpenAIWhisperModel by creating an OpenAI API client.
        """
        load_dotenv()
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            ).text
            return {"transcription": transcription}
