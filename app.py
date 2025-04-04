import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, request, jsonify, Response, send_file
import time
import torch
from generator import VoiceGenerator
import concurrent.futures
from functools import lru_cache
import shutil
from werkzeug.utils import secure_filename
import glob
from pydub import AudioSegment
import tempfile

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'resources'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}
MINIMUM_AUDIO_LENGTH = 30  # minimum length in seconds
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
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

# Create a single thread pool executor with optimal number of workers
executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=min(4, (os.cpu_count() or 1))
)

# Initialize generator once at startup
generator = VoiceGenerator()

# Cache reference speaker embeddings
@lru_cache(maxsize=32)
def get_cached_reference_speaker(reference_name):
    return f"resources/{reference_name}.mp3"

def make_response(status="ok", data=None, error=None, http_code=200):
    response = {
        "status": status,
        "timestamp": time.time()
    }
    if data is not None:
        response["data"] = data
    if error is not None:
        response["error"] = error
    return jsonify(response), http_code

@app.route('/generate-audio', methods=['POST'])
def generate_speech_endpoint():
    try:
        start_time = time.time()
        
        # Parse request data outside the task
        data = request.get_json()
        text = data.get('text')
        reference_name = data.get('reference_speaker')
        speed = float(data.get('speed', 1.0))
        
        if not text or not reference_name:
            return make_response(
                status="error",
                error="'text' and 'reference_speaker' are required",
                http_code=400
            )

        # Get cached reference speaker path
        reference_speaker = get_cached_reference_speaker(reference_name)
        
        # Submit task to thread pool
        future = executor.submit(
            generator.generate_speech,
            text, 
            reference_speaker,
            speed
        )
        
        # Set timeout to prevent hanging requests
        output_path = future.result(timeout=10)
        
        generation_time = time.time() - start_time
        print(f"Total request processing time: {generation_time:.2f} seconds")

        return send_file(
            output_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f'generated_speech_{int(time.time())}.wav'
        )

    except concurrent.futures.TimeoutError:
        return make_response(
            status="error",
            error="Request timed out",
            http_code=504
        )
    except Exception as e:
        print(f"Error in generate_speech: {e}")
        return make_response(
            status="error",
            error=str(e),
            http_code=500
        )

@app.route('/reference-voices', methods=['POST'])
def upload_reference_voice():
    """Upload a new reference voice file and convert to MP3"""
    try:
        if 'file' not in request.files:
            return make_response(
                status="error",
                error="No file provided",
                http_code=400
            )

        file = request.files['file']
        name = request.form.get('name')

        if not name:
            return make_response(
                status="error",
                error="Name is required",
                http_code=400
            )

        if file.filename == '':
            return make_response(
                status="error",
                error="No selected file",
                http_code=400
            )

        if not allowed_file(file.filename):
            return make_response(
                status="error",
                error=f"File type not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}",
                http_code=400
            )

        # Create a temporary file for the upload
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file.save(temp_file.name)
            
            # Check audio length
            if not check_audio_length(temp_file.name):
                # Clean up temp file
                os.unlink(temp_file.name)
                return make_response(
                    status="error",
                    error=f"Audio file must be at least {MINIMUM_AUDIO_LENGTH} seconds long",
                    http_code=400
                )
            
            # Define the final MP3 filename and path
            filename = secure_filename(f"{name}.mp3")
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            # Convert to MP3 format
            convert_to_mp3(temp_file.name, filepath)
            
            # Clean up the temporary file
            os.unlink(temp_file.name)

        # Clear the cache for this reference name
        if get_cached_reference_speaker.cache_info():
            get_cached_reference_speaker.cache_clear()

        return make_response(
            status="ok",
            data={
                "message": "Reference voice uploaded and converted to MP3 successfully",
                "filename": filename
            }
        )

    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file' in locals():
            try:
                os.unlink(temp_file.name)
            except:
                pass
        
        print(f"Error uploading reference voice: {e}")
        return make_response(
            status="error",
            error=str(e),
            http_code=500
        )

@app.route('/reference-voices', methods=['GET'])
def list_reference_voices():
    """List all available reference voices"""
    try:
        voices = []
        pattern = os.path.join(UPLOAD_FOLDER, "*.mp3")
        for filepath in glob.glob(pattern):
            filename = os.path.basename(filepath)
            name = filename.rsplit('.', 1)[0]
            voices.append({
                "name": name,
                "filename": filename,
                "format": "mp3"
            })

        return make_response(
            status="ok",
            data={
                "voices": voices
            }
        )

    except Exception as e:
        print(f"Error listing reference voices: {e}")
        return make_response(
            status="error",
            error=str(e),
            http_code=500
        )

@app.route('/reference-voices/<name>', methods=['GET'])
def download_reference_voice(name):
    """Download a specific reference voice file"""
    try:
        filepath = os.path.join(UPLOAD_FOLDER, f"{name}.mp3")
        if os.path.exists(filepath):
            return send_file(
                filepath,
                mimetype='audio/mpeg',
                as_attachment=True,
                download_name=os.path.basename(filepath)
            )

        return make_response(
            status="error",
            error="Reference voice not found",
            http_code=404
        )

    except Exception as e:
        print(f"Error downloading reference voice: {e}")
        return make_response(
            status="error",
            error=str(e),
            http_code=500
        )

@app.route('/reference-voices/<name>', methods=['DELETE'])
def delete_reference_voice(name):
    """Delete a specific reference voice file"""
    try:
        deleted = False
        for ext in ALLOWED_EXTENSIONS:
            filepath = os.path.join(UPLOAD_FOLDER, f"{name}.{ext}")
            if os.path.exists(filepath):
                os.remove(filepath)
                deleted = True
                # Clear the cache for this reference name
                if get_cached_reference_speaker.cache_info():
                    get_cached_reference_speaker.cache_clear()
                break

        if not deleted:
            return make_response(
                status="error",
                error="Reference voice not found",
                http_code=404
            )

        return make_response(
            status="ok",
            data={
                "message": "Reference voice deleted successfully"
            }
        )

    except Exception as e:
        print(f"Error deleting reference voice: {e}")
        return make_response(
            status="error",
            error=str(e),
            http_code=500
        )

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        return make_response(status="ok")
    except Exception as e:
        print(f"Health check failed: {e}")
        return make_response(
            status="error",
            error=str(e),
            http_code=500
        )

@app.route('/system-info', methods=['GET'])
def system_info():
    """Endpoint to check system configuration including GPU status"""
    if generator is None:
        return make_response(
            status="error",
            error="GPU with CUDA support is required but not available",
            http_code=503
        )

    try:
        device_info = {
            "device": generator.device,
            "cuda_available": torch.cuda.is_available()
        }
        return make_response(
            status="ok",
            data={"system_info": device_info}
        )
    except Exception as e:
        print(f"System info check failed: {e}")
        return make_response(
            status="error",
            error=str(e),
            http_code=500
        )

@app.route('/', methods=['GET'])
def index():
    """Simple landing page with API documentation"""
    api_docs = {
        "name": "Voice Generation API",
        "endpoints": {
            "/generate-audio": {
                "method": "POST",
                "content_type": "application/json",
                "description": "Generate speech from text with voice cloning",
                "parameters": {
                    "text": "Text to convert to speech",
                    "reference_speaker": "Name of the reference voice to use",
                    "speed": "(optional) Speech speed multiplier (default: 1.0)"
                }
            },
            "/reference-voices": {
                "method": "POST",
                "content_type": "multipart/form-data",
                "description": "Upload a new reference voice",
                "parameters": {
                    "file": "Audio file (mp3 or wav)",
                    "name": "Name for the reference voice"
                }
            },
            "/reference-voices": {
                "method": "GET",
                "description": "List all available reference voices"
            },
            "/reference-voices/<name>": {
                "method": "GET",
                "description": "Download a specific reference voice"
            },
            "/reference-voices/<name>": {
                "method": "DELETE",
                "description": "Delete a specific reference voice"
            },
            "/health": {
                "method": "GET",
                "description": "Health check endpoint"
            },
            "/system-info": {
                "method": "GET",
                "description": "System configuration and GPU status"
            }
        }
    }
    return make_response(status="ok", data=api_docs)

if __name__ == '__main__':
    # Use Gunicorn for production
    app.run(host='0.0.0.0', port=8585, debug=False)
