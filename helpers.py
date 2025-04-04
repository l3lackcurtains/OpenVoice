import os
import time
from flask import jsonify
from pydub import AudioSegment
from functools import lru_cache

# Configure constants
UPLOAD_FOLDER = 'resources'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}
MINIMUM_AUDIO_LENGTH = 30  # minimum length in seconds

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_mp3(input_path, output_path):
    """Convert any audio file to MP3 format"""
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format='mp3', bitrate='192k')

def check_audio_length(file_path):
    """Check if audio file meets minimum length requirement"""
    audio = AudioSegment.from_file(file_path)
    duration_seconds = len(audio) / 1000  # Convert milliseconds to seconds
    return duration_seconds >= MINIMUM_AUDIO_LENGTH

# Cache reference speaker embeddings
@lru_cache(maxsize=32)
def get_cached_reference_speaker(reference_name):
    """Get cached reference speaker path"""
    return f"{UPLOAD_FOLDER}/{reference_name}.mp3"

def make_response(status="ok", data=None, error=None, http_code=200):
    """Create a standardized JSON response"""
    response = {
        "status": status,
        "timestamp": time.time()
    }
    if data is not None:
        response["data"] = data
    if error is not None:
        response["error"] = error
    return jsonify(response), http_code

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)