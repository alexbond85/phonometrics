import numpy as np
import parselmouth  # type: ignore

from parselmouth.praat import call  # type: ignore


def extract_pitch(audio_data: np.ndarray, sample_rate: int):
    """
    Extract the pitch (fundamental frequency) from audio data.
    """
    # Ensure audio data is mono; if stereo, convert by averaging channels
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Create a Parselmouth Sound object from the audio data
    snd = parselmouth.Sound(values=audio_data, sampling_frequency=sample_rate)

    # Define pitch extraction parameters
    pitch_range = (75, 600)  # Standard pitch range for human voice
    pitch = call(snd, "To Pitch", 0.0, pitch_range[0], pitch_range[1])

    return pitch
