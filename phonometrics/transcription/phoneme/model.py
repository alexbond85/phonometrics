# mostly taken from https://github.com/jonatasgrosman/huggingsound
from __future__ import annotations
import torch
import torchaudio

from transformers import AutoModelForCTC
from transformers import AutoProcessor

from phonometrics.transcription.phoneme.decoder import GreedyDecoder
from phonometrics.transcription.phoneme.tokens import TokenSet


class Waveform:
    """Represents an audio waveform and its associated sample rate.

    Attributes
    ----------
    data : torch.Tensor
        The waveform data.
    sample_rate : int
        The sample rate of the waveform.
    """

    def __init__(self, data: torch.Tensor, sample_rate: int):
        """
        Parameters
        ----------
        data : torch.Tensor
            The waveform data.
        sample_rate : int
            The sample rate of the waveform.
        """
        self.data = data
        self.sample_rate = sample_rate

    def resample(self, target_sample_rate: int):
        """Resamples the waveform to the target sample rate if needed.

        Attention: This method modifies the waveform in place.

        Parameters
        ----------
        target_sample_rate : int
            The target sample rate.

        Returns
        -------
        Waveform
            The resampled waveform.
        """
        if self.sample_rate != target_sample_rate:
            self.data = torchaudio.functional.resample(
                self.data,
                orig_freq=self.sample_rate,
                new_freq=target_sample_rate,
            )
            self.sample_rate = target_sample_rate
        return self

    def to_mono(self) -> Waveform:
        """Converts the waveform to mono if it is stereo.

        Attention: This method modifies the waveform in place.

        Returns
        -------
        Waveform
            The mono waveform.
        """
        if self.data.shape[0] > 1:
            self.data = torch.mean(self.data, dim=0)
        return self


class TranscriptionModel:
    """Encapsulates the transcription model and related processing components.

    Attributes
    ----------
    _model : AutoModelForCTC
        The trained CTC model for transcription.
    _processor : AutoProcessor
        The processor for preparing audio inputs.
    _token_set : TokenSet
        The token set derived from the processor.
    _decoder : GreedyDecoder
        The decoder for generating transcriptions from logits.
    """

    def __init__(self, model: AutoModelForCTC, processor: AutoProcessor):
        """
        Parameters
        ----------
        model : AutoModelForCTC
            The trained CTC model for transcription.
        processor : AutoProcessor
            The processor for preparing audio inputs.
        """
        self._model = model
        self._processor = processor
        self._token_set = TokenSet.from_processor(processor)
        self._decoder = GreedyDecoder(self._token_set)

    def process_inputs(self, audio_waveform: torch.Tensor, sample_rate: int):
        """Prepares the audio input for the model.

        Parameters
        ----------
        audio_waveform : torch.Tensor
            The audio waveform data.
        sample_rate : int
            The sample rate of the audio.

        Returns
        -------
        dict
            The processed input ready for model inference.
        """
        return self._processor(
            audio_waveform.squeeze(),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

    def infer(self, inputs):
        """Performs inference using the model.

        Parameters
        ----------
        inputs : dict
            The processed input tensor.

        Returns
        -------
        torch.Tensor
            The logits produced by the model.
        """
        with torch.no_grad():
            logits = self._model(inputs.input_values).logits
        return logits

    def decode(self, logits):
        """Decodes the model's logits into a transcription.

        Parameters
        ----------
        logits : torch.Tensor
            The logits output from the model.

        Returns
        -------
        str
            The decoded transcription.
        """
        return self._decoder(logits)[0]


class AudioTranscriber:
    """Handles the audio transcription process.

    Attributes
    ----------
    _transcription_model : TranscriptionModel
        The model used for processing and transcription.
    """

    def __init__(self, transcription_model: TranscriptionModel):
        """
        Parameters
        ----------
        transcription_model : TranscriptionModel
            The model used for processing and transcription.
        """
        self._transcription_model = transcription_model

    def transcribe(self, waveform: Waveform):
        """Transcribes the given audio waveform.

        Parameters
        ----------
        waveform : Waveform
            The audio waveform to be transcribed.

        Returns
        -------
        str
            The transcription of the audio.
        """
        waveform.resample(16000).to_mono()
        inputs = self._transcription_model.process_inputs(
            waveform.data, waveform.sample_rate
        )
        logits = self._transcription_model.infer(inputs)
        return self._transcription_model.decode(logits)

