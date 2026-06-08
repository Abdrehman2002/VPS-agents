FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY agent.py .

# Pre-download plugin model files (turn detector, Silero VAD) so the first call is instant
RUN python agent.py download-files || true

# Production worker (connects to LIVEKIT_URL from env, dispatched by agent_name)
CMD ["python", "agent.py", "start"]
