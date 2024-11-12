import yaml
import os
import re
import requests
import io
import numpy as np
import gradio as gr
import soundfile as sf
from phonometrics.vizualize.plot import WaveformPlotBuilder
from phonometrics.audio_processing.pitch import extract_pitch


class AppConfig:

    def __init__(self, config_content):
        self.config_content = config_content

    def audio_files_path(self):
        return self.config_content.get("audio_folder", "audio_folder")

    def phoneme_transcription_url(self):
        return self.config_content.get("transcription_service_url")

    def word_transcription_url(self):
        # Use the URLs from the configuration
        phoneme_transcription_service_url = self.config_content.get("transcription_service_url")
        preferred_transcription_service = self.config_content.get("preferred_word_transcription_service", "local")
        if preferred_transcription_service == "local":
            return self.config_content.get("local_word_transcription_service_url")
        else:
            return self.config_content.get("openai_word_transcription_service_url")

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
app_config = AppConfig(config)

def list_audio_files(folder_path: str):
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))]
    audio_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    return audio_files


def process_selected_file(selected_file):
    # Read the audio file
    file_path = os.path.join(app_config.audio_files_path(), selected_file)
    with open(file_path, 'rb') as f:
        audio_bytes = f.read()

    transcription_phonemes = transcribe_phonemes(audio_bytes)
    transcription_words = transcribe_words(audio_bytes)
    # Load audio data using soundfile
    audio_file = io.BytesIO(audio_bytes)
    y, sr = sf.read(audio_file)

    # Convert to mono if necessary
    if y.ndim > 1:
        y = y.mean(axis=1)
    phonemes = transcription_phonemes['transcription']
    img_array = plot_waveform_and_pitch(y, sr, transcription_phonemes)
    words = transcription_words['transcription']
    # Return audio file path, transcriptions, and image array
    return file_path, phonemes, words, img_array


def transcribe_phonemes(audio_bytes):
    # Prepare the files payload
    files = {
        'file': ('recorded_audio.wav', audio_bytes, 'audio/wav')
    }
    # Send the audio to the phoneme transcription service
    try:
        response_phonemes = requests.post(app_config.phoneme_transcription_url(), files=files)
        response_phonemes.raise_for_status()
        transcription_phonemes = response_phonemes.json()
    except requests.exceptions.RequestException as e:
        print(f"Error contacting phoneme transcription service: {e}")
        transcription_phonemes = {'transcription': ''}
    return transcription_phonemes


def transcribe_words(audio_bytes):
    # Prepare the files payload
    files = {
        'file': ('recorded_audio.wav', audio_bytes, 'audio/wav')
    }
    # Send the audio to the word transcription service
    try:
        response_words = requests.post(app_config.word_transcription_url(), files=files)
        response_words.raise_for_status()
        transcription_words = response_words.json()
    except requests.exceptions.RequestException as e:
        print(f"Error contacting word transcription service: {e}")
        transcription_words = {'transcription': ''}
    return transcription_words


def plot_waveform_and_pitch(y, sr, phonemes_transcription):
    # Extract pitch
    pitch = extract_pitch(y, sr)

    # Draw waveform, pitch, and transcription
    fig = (WaveformPlotBuilder("Recorded")
           .with_pitch(pitch)
           .with_transcription(phonemes_transcription)
           .with_waveform(y, sr)
           .build())

    # Convert Matplotlib figure to image array
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img_array

# Get the list of audio files once and store it
audio_files = list_audio_files(app_config.audio_files_path())

with gr.Blocks() as demo:
    gr.Markdown('# Accent Reduction App')
    gr.Markdown('Navigate through audio files or record your voice for analysis and practice.')

    # Initialize current index and reference transcriptions as state variables
    current_index = gr.State(0)
    reference_phoneme_transcription = gr.State('')
    reference_word_transcription = gr.State('')

    with gr.Row():
        with gr.Column():
            gr.Markdown('## 1. Navigate Audio Files')

            # Create UI components
            selected_file = gr.Dropdown(choices=audio_files, label='Select an audio file')

            # Place buttons side by side
            with gr.Row():
                prev_button = gr.Button('Previous')
                analyse_reference_button = gr.Button('Analyze Reference')
                next_button = gr.Button('Next')

            audio_player = gr.Audio(label='Audio', type='filepath')
            transcription_text_phonemes = gr.Textbox(label='Phoneme Transcription')
            transcription_text_words = gr.Textbox(label='Word Transcription')
            plot_image = gr.Image(type='numpy', label='Waveform and Pitch')

            # Function to handle "Previous" button click
            def on_prev_button_click(current_index):
                current_index = max(0, current_index - 1)
                selected_file_value = audio_files[current_index]
                results = process_selected_file(selected_file_value)
                file_path, transcription_phonemes, transcription_words, img_array = results
                return [current_index, selected_file_value, file_path, transcription_phonemes, transcription_words, img_array, transcription_phonemes]

            # Function to handle "Next" button click
            def on_next_button_click(current_index):
                current_index = min(len(audio_files) - 1, current_index + 1)
                selected_file_value = audio_files[current_index]
                results = process_selected_file(selected_file_value)
                file_path, transcription_phonemes, transcription_words, img_array = results
                return [current_index, selected_file_value, file_path, transcription_phonemes, transcription_words, img_array, transcription_phonemes]

            # Function to handle "Play and Analyze" button click
            def on_analyse_reference_button_click(selected_file_value):
                current_index = audio_files.index(selected_file_value)
                results = process_selected_file(selected_file_value)
                file_path, transcription_phonemes, transcription_words, img_array = results
                return [current_index, file_path, transcription_phonemes, transcription_words, img_array, transcription_phonemes]

            # Function to handle dropdown selection change
            def on_selected_file_change(selected_file_value):
                current_index = audio_files.index(selected_file_value)
                return current_index

            # Set up event handlers
            prev_button.click(
                on_prev_button_click,
                inputs=current_index,
                outputs=[current_index, selected_file, audio_player, transcription_text_phonemes, transcription_text_words, plot_image]
            )

            next_button.click(
                on_next_button_click,
                inputs=current_index,
                outputs=[current_index, selected_file, audio_player, transcription_text_phonemes, transcription_text_words, plot_image]
            )

            analyse_reference_button.click(
                on_analyse_reference_button_click,
                inputs=selected_file,
                outputs=[current_index, audio_player, transcription_text_phonemes, transcription_text_words, plot_image]
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
            rec_transcription_text_phonemes = gr.Textbox(label='Phoneme Transcription')
            rec_transcription_text_words = gr.Textbox(label='Word Transcription')
            rec_plot_image = gr.Image(type='numpy', label='Waveform and Pitch')
            comparison_result_text = gr.HTML(label='Phonetic Transcription Comparison')

            def on_analyze_recorded_button_click(audio):
                if not audio:
                    return
                sr, y = audio  # Get the sample rate and audio data
                if y.ndim > 1:
                    y = y.mean(axis=1)
                    # Write it to bytes
                buf = io.BytesIO()
                sf.write(buf, y, sr, format='WAV')
                buf.seek(0)  # Reset buffer position

                audio_bytes = buf.read()
                # Transcribe phonemes
                user_transcription_phonemes = transcribe_phonemes(audio_bytes)
                phonemes = user_transcription_phonemes['transcription']
                user_transcription_words = transcribe_words(audio_bytes)
                words = user_transcription_words['transcription']
                img_array = plot_waveform_and_pitch(y, sr, user_transcription_phonemes)
                # Compare phonetic transcriptions
                return phonemes, words, img_array

            analyze_button.click(
                on_analyze_recorded_button_click,
                inputs=[record_audio],
                outputs=[rec_transcription_text_phonemes, rec_transcription_text_words, rec_plot_image]
            )

demo.launch()
