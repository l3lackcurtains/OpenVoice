FROM nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential ffmpeg git \
    python3 python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

COPY . .

# Install Python dependencies in fewer layers
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install git+https://github.com/myshell-ai/MeloTTS.git && \
    python3 -m unidic download && \
    pip3 install -r requirements.txt


RUN bash ./setup_cuda.sh

RUN python3 download.py

EXPOSE 8585

CMD ["python3", "app.py"]
