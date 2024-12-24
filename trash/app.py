from flask import Flask, jsonify, request, send_from_directory, render_template
import numpy as np
from pydub import AudioSegment
import parselmouth
import os

app = Flask(__name__, static_folder=".", template_folder=".")

# Temporary folder to store uploaded files
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def load_audio_mp3(file_path):
    audio = AudioSegment.from_mp3(file_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(44100)  # Set sample rate to 44100 Hz
    samples = np.array(audio.get_array_of_samples(), dtype="float64") / (2**15)  # Normalize
    return samples, audio.frame_rate


import math


def extract_formants(file_path):
    samples, sample_rate = load_audio_mp3(file_path)
    sound = parselmouth.Sound(samples, sampling_frequency=sample_rate)
    formant = sound.to_formant_burg()

    times = np.arange(0, formant.xmax, 0.01)

    # Replace NaN values with None
    f1 = [formant.get_value_at_time(1, t) for t in times]
    f1 = [None if (val is None or math.isnan(val)) else val for val in f1]

    f2 = [formant.get_value_at_time(2, t) for t in times]
    f2 = [None if (val is None or math.isnan(val)) else val for val in f2]

    return {"times": times.tolist(), "f1": f1, "f2": f2}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    formant_data = extract_formants(file_path)
    return jsonify(formant_data)

@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)