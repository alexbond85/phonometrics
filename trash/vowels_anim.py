import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from pydub import AudioSegment
import parselmouth
import sounddevice as sd
import threading

def load_audio_mp3(file_path):
    """
    Load an MP3 file and convert it to a format suitable for Parselmouth and playback.
    """
    audio = AudioSegment.from_mp3(file_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(44100)  # Set sample rate to 44100 Hz
    samples = np.array(audio.get_array_of_samples(), dtype="float64") / (2**15)  # Normalize
    return samples, audio.frame_rate

def extract_formants(sound):
    """
    Extract formant frequencies using Parselmouth.
    """
    return sound.to_formant_burg()

def play_audio(audio_data, sample_rate):
    """
    Play the audio data using sounddevice.
    """
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()

def update_plot(frame, times, formant, f1_line, f2_line):
    """
    Update the plot dynamically during audio playback.
    """
    if frame >= len(times):
        return f1_line, f2_line

    current_times = times[:frame]
    f1_values = [formant.get_value_at_time(1, t) for t in current_times]
    f2_values = [formant.get_value_at_time(2, t) for t in current_times]

    f1_line.set_data(current_times, f1_values)
    f2_line.set_data(current_times, f2_values)

    return f1_line, f2_line

def on_play_clicked(event, audio_data, sample_rate, times, formant, fig, f1_line, f2_line):
    """
    Handle the play button click event.
    """
    def play_and_plot():
        # Play audio in a separate thread to avoid blocking
        threading.Thread(target=play_audio, args=(audio_data, sample_rate)).start()

        # Update the plot while the audio is playing
        for frame in range(1, len(times)):
            update_plot(frame, times, formant, f1_line, f2_line)
            plt.pause(0.01)  # Adjust the pause for smooth animation

    play_and_plot()

def main(file_path):
    # Load the audio
    audio_data, sample_rate = load_audio_mp3(file_path)
    sound = parselmouth.Sound(audio_data, sampling_frequency=sample_rate)
    formant = extract_formants(sound)

    # Get times for analysis
    times = np.arange(0, formant.xmax, 0.01)

    # Setup the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, formant.xmax)
    ax.set_ylim(0, 4000)  # Typical F1/F2 frequency range in Hz
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Formant Frequencies Over Time")
    ax.grid(True)

    f1_line, = ax.plot([], [], label="F1 (Hz)", color="blue")
    f2_line, = ax.plot([], [], label="F2 (Hz)", color="orange", linestyle="--")
    ax.legend()

    # Add play button
    play_button_ax = plt.axes([0.4, 0.01, 0.2, 0.05])  # Button position
    play_button = Button(play_button_ax, "Play")

    # Connect play button
    play_button.on_clicked(lambda event: on_play_clicked(event, audio_data, sample_rate, times, formant, fig, f1_line, f2_line))

    # Display the plot
    plt.show()

if __name__ == "__main__":
    file_path = "i.mp3"  # Replace with your MP3 file path
    main(file_path)