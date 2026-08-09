# LUCID-PD CDSS — single container for a Hugging Face Docker Space.
#
# Stage 1 compiles the React bundle; stage 2 runs the FastAPI service and serves that
# bundle from the same origin, so the whole system is one URL on one host. Torch is
# installed from the CPU wheel index because Spaces hardware for this deployment has no
# GPU and the default wheels would pull in several gigabytes of unused CUDA libraries.

FROM node:20-slim AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim

# ffmpeg transcodes browser recordings (WebM/Opus) to WAV; libsndfile backs the readers
# used by openSMILE for the formats that need no transcoding.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000; everything the app writes lives under its home.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/home/user/app/src \
    STATIC_DIR=/home/user/app/static \
    TORCH_HOME=/home/user/.cache/torch \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    HF_HOME=/home/user/.cache/huggingface \
    SLM_BACKEND=hf
WORKDIR /home/user/app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.9.1 torchvision
COPY --chown=user:user requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Cache the ImageNet ResNet-50 weights at build time. build_model() rebuilds the
# architecture from the pretrained checkpoint before loading the fine-tuned state dict,
# so without this the first MRI request would wait on a download.
RUN python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.DEFAULT)"

COPY --chown=user:user src/ src/
COPY --chown=user:user app/ app/
COPY --chown=user:user models/ models/
COPY --chown=user:user data/ data/
COPY --from=ui --chown=user:user /ui/dist/ static/

EXPOSE 7860
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
