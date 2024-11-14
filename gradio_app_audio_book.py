from fileinput import filename

import gradio as gr
from io import BytesIO
from pydub import AudioSegment
import os
import openai  # Make sure to install the OpenAI library: pip install openai
from dotenv import load_dotenv  # For loading API key from .env file

# Load API key from .env file
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
client = openai.OpenAI(api_key=openai_api_key)

def parse_srt_file(srt_file_path):
    # [Parsing function remains the same]
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
        return f"Phrase {index}: **{sub['text']}**", audio_bytes  # Return text, audio
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
chapter = '02-Chapitre_2'
srt_file_path = chapter + '.srt'
audio_file_path = chapter + '.mp3'


subtitles = parse_srt_file(srt_file_path)

# Load the audio file once at the beginning

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
    #phrase_display = gr.Markdown("<p style='font-size: 6em; font-weight: bold; color: #333333;'>",
    #                             elem_id="phrase_display")
    with gr.Row():
        prev_button = gr.Button("Previous")
        next_button = gr.Button("Next")

    # Remove 'continuous_update' parameter
    phrase_slider = gr.Slider(1, len(subtitles), value=1, step=1, label="Select Phrase", interactive=True)

    # Use gr.State to store the phrase index internally
    phrase_index = gr.State(value=1)

    # Set autoplay=True for automatic audio playback
    audio_output = gr.Audio(label="Audio Playback", elem_id="audio_output", format='mp3', autoplay=True)

    # Add a "Translate" button
    translate_button = gr.Button("Translate")

    # Add a component to display the translation
    translation_display = gr.Markdown("", elem_id="translation_display")

    # Update phrase and audio when index changes
    def update_phrase_and_audio(index):
        text, audio_bytes = get_audio_chunk_and_phrase(index)
        return text, audio_bytes, index, index, ""  # Reset translation when phrase changes

    # Function to handle translation using OpenAI ChatCompletion API
    def translate_phrase(phrase_text):
        # Ensure the API key is set
        if not client.api_key:
            return "OpenAI API key not set. Please set your API key."

        # Construct the messages for the conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant that translates French to Russian and explains words that might be unfamiliar to a Russian learner. You speak only russian and french."},
            {"role": "user", "content": f"""Translate the following French phrase into Russian, and then list the words that a Russian learner might not know, providing short translations for each, don't translate all words, only difficult ones. for example: Se promener - Гулять.
            

Phrase: "{phrase_text}"."""}
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=1250,
                temperature=0.5,
            )
            translation = response.choices[0].message.content
        except Exception as e:
            translation = f"Error in translation: {e}"

        return translation

    # Connect components
    outputs = [phrase_display, audio_output, phrase_index, phrase_slider, translation_display]

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

    # Handle translation button click
    translate_button.click(fn=lambda text: translate_phrase(text), inputs=phrase_display, outputs=translation_display)

demo.launch(server_name="0.0.0.0", server_port=7860)
