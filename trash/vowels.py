import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import parselmouth

def load_audio_mp3(file_path):
    """
    Load an MP3 file and convert it to a format suitable for Parselmouth.
    """
    audio = AudioSegment.from_mp3(file_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(44100)  # Set sample rate to 44100 Hz
    samples = np.array(audio.get_array_of_samples(), dtype="float64") / (2**15)  # Normalize
    return parselmouth.Sound(samples, sampling_frequency=audio.frame_rate)

def extract_formants(sound):
    """
    Extract formant frequencies using Parselmouth.
    """
    return sound.to_formant_burg()

def get_f1_f2_over_time(formant):
    """
    Get F1 and F2 values over time.
    """
    times = np.arange(0, formant.xmax, 0.01)  # Analyze every 10 ms
    f1_values = [formant.get_value_at_time(1, t) for t in times]
    f2_values = [formant.get_value_at_time(2, t) for t in times]
    return times, f1_values, f2_values

def plot_f1_f2(times, f1_values, f2_values):
    """
    Plot F1 and F2 formant frequencies over time.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(times, f1_values, label="F1 (Hz)", color="blue")
    plt.plot(times, f2_values, label="F2 (Hz)", color="orange", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Formant Frequencies Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file_path = "i.mp3"  # Replace with your MP3 file path

    # Load and process the MP3 audio file
    sound = load_audio_mp3(file_path)

    # Extract formants
    formant = extract_formants(sound)

    # Get F1 and F2 values over time
    times, f1_values, f2_values = get_f1_f2_over_time(formant)

    # Plot the results
    plot_f1_f2(times, f1_values, f2_values)