import gradio as gr
from io import BytesIO
from pydub import AudioSegment
import os

def parse_srt_file(srt_file_path):
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
        return "", None  # Return empty text, no audio

    sub = subtitles[index - 1]
    start_time = sub['start_time'] * 1000  # Convert to milliseconds
    end_time = sub['end_time'] * 1000

    try:
        # Check if audio chunk is already in cache
        if index in audio_chunks:
            audio_bytes = audio_chunks[index]
        else:
            if audio is None:
                return sub['text'], None

            # Extract the chunk
            chunk = audio[start_time:end_time]

            # Export to raw bytes
            audio_buffer = BytesIO()
            chunk.export(audio_buffer, format='mp3')
            audio_bytes = audio_buffer.getvalue()

            # Store audio_bytes for future use
            audio_chunks[index] = audio_bytes

        # Return text and audio_bytes
        return f"Phrase {index}: {sub['text']}", audio_bytes  # Return text, audio
    except Exception as e:
        print(f"Error processing audio chunk: {e}")
        return sub['text'], None

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
srt_file_path = 'premiere-partie.srt'
subtitles = parse_srt_file(srt_file_path)

# Load the audio file once at the beginning
audio_file_path = 'premiere-partie.mp3'
if not os.path.exists(audio_file_path):
    print(f"Audio file not found at path: {audio_file_path}")
    audio = None
else:
    audio = AudioSegment.from_mp3(audio_file_path)
    print(f"Loaded audio file: {audio_file_path}")

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

    # Set continuous_update=False to prevent continuous loading
    phrase_slider = gr.Slider(1, len(subtitles), value=1, step=1, label="Select Phrase", interactive=True)

    # Use gr.State to store the phrase index internally
    phrase_index = gr.State(value=1)

    # Set autoplay=True for automatic audio playback
    audio_output = gr.Audio(label="Audio Playback", elem_id="audio_output", format='mp3', autoplay=True)

    # Update phrase and audio when index changes
    def update_phrase_and_audio(index):
        text, audio_bytes = get_audio_chunk_and_phrase(index)
        return text, audio_bytes, index, index

    # Connect components
    outputs = [phrase_display, audio_output, phrase_index, phrase_slider]

    # Update phrase and audio when slider value changes
    phrase_slider.change(update_phrase_and_audio, inputs=phrase_slider, outputs=outputs)

    # Functions to handle button clicks
    def decrement_and_update(idx):
        new_index = decrement_index(idx)
        return update_phrase_and_audio(new_index)

    def increment_and_update(idx):
        new_index = increment_index(idx)
        return update_phrase_and_audio(new_index)

    # Bind buttons to functions
    prev_button.click(fn=decrement_and_update, inputs=phrase_index, outputs=outputs)
    next_button.click(fn=increment_and_update, inputs=phrase_index, outputs=outputs)

demo.launch(server_name="0.0.0.0", server_port=7860)
