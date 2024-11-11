from abc import ABC
from abc import abstractmethod
from typing import Dict


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
