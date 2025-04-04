import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, request, jsonify, Response, send_file
import time
import torch
from generator import VoiceGenerator
import concurrent.futures
from functools import lru_cache

app = Flask(__name__)

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
    # Use Gunicorn for production
    app.run(host='0.0.0.0', port=8585, debug=False)
