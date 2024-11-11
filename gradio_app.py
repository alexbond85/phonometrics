import yaml
import os
import re
import mimetypes
import requests
import io
import numpy as np
import gradio as gr
import soundfile as sf
from phonometrics.vizualize.plot import WaveformPlotBuilder
from phonometrics.audio_processing.pitch import extract_pitch

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Use the URLs from the configuration
phoneme_transcription_service_url = config.get("transcription_service_url")
audio_folder = config.get("audio_folder", "frenchphrases/output_segments")  # Replace with your folder path


def get_audio_files():
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))]
    audio_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    return audio_files


def process_audio_file(selected_file):
    # Read the audio file
    file_path = os.path.join(audio_folder, selected_file)
    with open(file_path, 'rb') as f:
        audio_bytes = f.read()

    # Prepare the files payload
    files = {
        'file': (selected_file, audio_bytes, 'audio/wav')
    }

    # Send the audio file to the phoneme transcription service
    try:
        response = requests.post(phoneme_transcription_service_url, files=files)
        response.raise_for_status()
        transcription = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error contacting transcription service: {e}")
        transcription = {'transcription': ''}

    # Load audio data using soundfile
    audio_file = io.BytesIO(audio_bytes)
    y, sr = sf.read(audio_file)

    # Convert to mono if necessary
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Extract pitch
    pitch = extract_pitch(y, sr)

    # Draw waveform, pitch, and transcription

    fig = (WaveformPlotBuilder("Reference")
           .with_pitch(pitch)
           .with_transcription(transcription)
           .with_waveform(y, sr)
           .build())

    # Convert Matplotlib figure to image array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Return audio file path, transcription, and image array
    return file_path, transcription.get('transcription', ''), img_array


def process_recorded_audio(audio):
    y, sr = audio
    print(f"Recorded audio: {y.shape[0]} samples, {sr} Hz")

    # Write it to bytes
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV')
    buf.seek(0)  # Reset buffer position

    audio_bytes = buf.read()

    # Prepare the files payload
    files = {
        'file': ('recorded_audio.wav', audio_bytes, 'audio/wav')
    }

    # Send the audio to the phoneme transcription service
    try:
        response = requests.post(phoneme_transcription_service_url, files=files)
        response.raise_for_status()
        transcription = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error contacting transcription service: {e}")
        transcription = {'transcription': ''}

    # Extract pitch
    pitch = extract_pitch(y, sr)

    # Draw waveform, pitch, and transcription
    fig = (WaveformPlotBuilder("Recorded Voice")
              .with_pitch(pitch)
           .with_transcription(transcription)
              .with_waveform(y, sr)
                .build())
    # Convert Matplotlib figure to image array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Return transcription and image array
    return transcription.get('transcription', ''), img_array


# Get the list of audio files once and store it
audio_files = get_audio_files()

with gr.Blocks() as demo:
    gr.Markdown('# Audio Navigation and Record App')
    gr.Markdown('Navigate through audio files or record your voice for analysis and practice.')

    # Initialize current index as a state variable
    current_index = gr.State(0)

    with gr.Row():
        with gr.Column():
            gr.Markdown('## 1. Navigate Audio Files')

            # Create UI components
            selected_file = gr.Dropdown(choices=audio_files, label='Select an audio file')

            # Place buttons side by side
            with gr.Row():
                prev_button = gr.Button('Previous')
                play_button = gr.Button('Play and Analyze')
                next_button = gr.Button('Next')

            audio_player = gr.Audio(label='Audio', type='filepath')
            transcription_text = gr.Textbox(label='Transcription from Service')
            plot_image = gr.Image(type='numpy', label='Waveform and Pitch')

            # Function to handle "Previous" button click
            def on_prev_button_click(current_index):
                current_index = max(0, current_index - 1)
                selected_file_value = audio_files[current_index]
                results = process_audio_file(selected_file_value)
                return [current_index, selected_file_value] + list(results)

            # Function to handle "Next" button click
            def on_next_button_click(current_index):
                current_index = min(len(audio_files) - 1, current_index + 1)
                selected_file_value = audio_files[current_index]
                results = process_audio_file(selected_file_value)
                return [current_index, selected_file_value] + list(results)

            # Function to handle "Play and Analyze" button click
            def on_play_button_click(selected_file_value):
                current_index = audio_files.index(selected_file_value)
                results = process_audio_file(selected_file_value)
                return [current_index] + list(results)

            # Function to handle dropdown selection change
            def on_selected_file_change(selected_file_value):
                current_index = audio_files.index(selected_file_value)
                return current_index

            # Set up event handlers
            prev_button.click(
                on_prev_button_click,
                inputs=current_index,
                outputs=[current_index, selected_file, audio_player, transcription_text, plot_image]
            )

            next_button.click(
                on_next_button_click,
                inputs=current_index,
                outputs=[current_index, selected_file, audio_player, transcription_text, plot_image]
            )

            play_button.click(
                on_play_button_click,
                inputs=selected_file,
                outputs=[current_index, audio_player, transcription_text, plot_image]
            )

            selected_file.change(
                on_selected_file_change,
                inputs=selected_file,
                outputs=current_index
            )

        with gr.Column():
            gr.Markdown('## 2. Record Your Voice')
            gr.Markdown("Click the 'Record' button to record your voice.")

            # Use 'type' parameter for gr.Audio
            record_audio = gr.Audio(type='numpy')
            analyze_button = gr.Button('Analyze Recording')
            rec_transcription_text = gr.Textbox(label='Transcription from Service')
            rec_plot_image = gr.Image(type='numpy', label='Waveform and Pitch')

            def on_analyze_button_click(audio):
                results = process_recorded_audio(audio)
                return results

            analyze_button.click(
                on_analyze_button_click,
                inputs=record_audio,
                outputs=[rec_transcription_text, rec_plot_image]
            )

demo.launch()