import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, request, jsonify, Response, send_file
from io import BytesIO
import time

import torch
from generator import VoiceGenerator
import threading
from queue import Queue
import concurrent.futures
import soundfile as sf

app = Flask(__name__)
request_queue = Queue()
thread_lock = threading.Lock()

try:
    generator = VoiceGenerator()
except RuntimeError as e:
    print(f"Failed to initialize VoiceGenerator: {e}")
    generator = None

# Create a thread pool executor
executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

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

def generate_speech_task(text, reference_speaker, speed):
    with thread_lock:
        return generator.generate_speech(text, reference_speaker, speed)

@app.route('/generate-audio', methods=['POST'])
def generate_speech_endpoint():
    try:
        start_time = time.time()
        print("Processing generate_speech request")
        
        data = request.get_json()
        text = data.get('text')
        reference_name = data.get('reference_speaker')
        speed = float(data.get('speed', 1.0))
        
        if not text:
            return make_response(
                status="error",
                error="'text' is required",
                http_code=400
            )
            
        if not reference_name:
            return make_response(
                status="error",
                error="'reference_speaker' is required",
                http_code=400
            )

        reference_speaker = f"resources/{reference_name}.mp3"
        generation_start = time.time()
        
        # Get the output path from the generator
        future = executor.submit(generate_speech_task, text, reference_speaker, speed)
        output_path = future.result()  # This should return the output file path
        
        generation_time = time.time() - generation_start
        print(f"Total request processing time: {time.time() - start_time:.2f} seconds")

        return send_file(
            output_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name=f'generated_speech_{int(time.time())}.wav'
        )

    except Exception as e:
        print(f"Error in generate_speech: {e}")
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
            "/generate": {
                "method": "POST",
                "content_type": "application/json",
                "description": "Generate speech from text with voice cloning",
                "parameters": {
                    "text": "Text to convert to speech",
                    "reference_speaker": "Path to reference audio file for voice cloning",
                    "speed": "(optional) Speech speed multiplier (default: 1.0)"
                }
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
    app.run(host='0.0.0.0', port=8585, debug=False, threaded=True)
