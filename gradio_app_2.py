import gradio as gr
from io import BytesIO
from pydub import AudioSegment
import os
import requests  # Import the requests library

def parse_srt_file(srt_file_path):
    # [The parse_srt_file function remains the same as before]
    subtitles = []
    try:
        with open(srt_file_path, 'r', encoding='utf-8') as file:
            srt_text = file.read()
    except Exception as e:
        print(f"Error reading SRT file: {e}")
        return subtitles

    entries = srt_text.strip().split('\n\n')
    for entry in entries:
        lines = entry.strip().split('\n')
        if len(lines) >= 3:
            try:
                index = int(lines[0])
                times = lines[1]
                text = ' '.join(lines[2:])
                start_str, end_str = times.split(' --> ')

                def time_to_seconds(time_str):
                    h, m, s_ms = time_str.strip().split(':')
                    s, ms = s_ms.split(',')
                    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0

                start_time = time_to_seconds(start_str)
                end_time = time_to_seconds(end_str)
                subtitles.append({
                    'index': index,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text
                })
            except Exception as e:
                print(f"Error parsing entry:\n{entry}\nError: {e}")
                continue
    return subtitles

# Global dictionary to store audio chunks for future use
audio_chunks = {}

def get_audio_chunk_and_phrase(index):
    index = int(index)
    if index < 1 or index > len(subtitles):
        print(f"Invalid index: {index}")
        return "", None, ""  # Return empty text, no audio, empty transcription

    sub = subtitles[index - 1]
    start_time = sub['start_time'] * 1000  # Convert to milliseconds
    end_time = sub['end_time'] * 1000

    try:
        # Load the audio file
        audio_file_path = os.path.join('ENFR-F1-GSR-DAY101.mp3')
        audio_file_path = os.path.join('premiere-partie.mp3')
        if not os.path.exists(audio_file_path):
            print(f"Audio file not found at path: {audio_file_path}")
            return sub['text'], None, ""
        audio = AudioSegment.from_mp3(audio_file_path)

        # Extract the chunk
        chunk = audio[start_time:end_time]

        # Export to raw bytes
        audio_buffer = BytesIO()
        chunk.export(audio_buffer, format='mp3')
        audio_bytes = audio_buffer.getvalue()

        # Store BytesIO for future use
        audio_buffer.seek(0)  # Reset buffer position
        audio_chunks[index] = audio_buffer  # Store the BytesIO object

        # Send audio chunk to the endpoint and get transcription
        transcription = {"transcription": "dummy"}#transcribe_audio(audio_buffer)

        return f"Phrase {index}: {sub['text']}", audio_bytes, transcription  # Return text, audio, transcription
    except Exception as e:
        print(f"Error processing audio chunk: {e}")
        return sub['text'], None, ""

def transcribe_audio(audio_buffer):
    try:
        # Prepare the files dictionary for the POST request
        files = {
            'file': ('audio.mp3', audio_buffer, 'audio/mpeg')
        }
        # Send the POST request to the endpoint
        response = requests.post('http://localhost:8000/transcribe/phonemes', files=files)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Parse the JSON response
        json_response = response.json()
        transcription = json_response.get('transcription', '')
        return transcription
    except Exception as e:
        print(f"Error in transcribe_audio: {e}")
        return "Error in transcription"

def increment_index(current_index):
    new_index = int(current_index) + 1
    if new_index > len(subtitles):
        new_index = len(subtitles)
    return new_index

def decrement_index(current_index):
    new_index = int(current_index) - 1
    if new_index < 1:
        new_index = 1
    return new_index

# Read subtitles from the file
srt_file_path = 'ENFR-F1-GSR-DAY101.srt'
srt_file_path = 'premiere-partie.srt'

subtitles = parse_srt_file(srt_file_path)

if not subtitles:
    print("No subtitles found. Please check the SRT file.")
else:
    print(f"Loaded {len(subtitles)} subtitles.")

with gr.Blocks() as demo:
    gr.Markdown("# Phrase Navigator")
    phrase_display = gr.Markdown("", elem_id="phrase_display")

    with gr.Row():
        prev_button = gr.Button("Previous")
        next_button = gr.Button("Next")
        phrase_index = gr.Number(value=1, label="Phrase Number", interactive=True)

    phrase_slider = gr.Slider(1, len(subtitles), value=1, step=1, label="Select Phrase", interactive=True)

    # Specify type='bytes' and format='mp3' for the audio component
    audio_output = gr.Audio(label="Audio Playback", elem_id="audio_output", format='mp3')

    # Add a component to display the transcription
    transcription_display = gr.Textbox(label="Transcription", interactive=False)

    # Update phrase, audio, and transcription when index changes
    def update_phrase_and_audio(index):
        text, audio, transcription = get_audio_chunk_and_phrase(index)
        return text, audio, transcription, index, index

    # Connect components
    outputs = [phrase_display, audio_output, transcription_display, phrase_index, phrase_slider]

    phrase_index.change(update_phrase_and_audio, inputs=phrase_index, outputs=outputs)
    phrase_slider.change(update_phrase_and_audio, inputs=phrase_slider, outputs=outputs)

    prev_button.click(fn=lambda idx: decrement_index(idx), inputs=phrase_index, outputs=phrase_index)
    next_button.click(fn=lambda idx: increment_index(idx), inputs=phrase_index, outputs=phrase_index)

    # Automatically update when prev/next buttons are clicked
    phrase_index.change(update_phrase_and_audio, inputs=phrase_index, outputs=outputs)

demo.launch(server_name="0.0.0.0", server_port=7860)
