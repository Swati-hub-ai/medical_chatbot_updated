# Use a lightweight official Python base image
FROM python:3.11-slim

# Avoid writing .pyc files & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Accept HuggingFace token securely
ARG HF_TOKEN
ENV HF_TOKEN=$HF_TOKEN

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Cloud Run expects the app to listen on port 8080
EXPOSE 8080

# ✅ Streamlit server configuration via env vars
ENV PORT 8080
ENV STREAMLIT_SERVER_PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Run the Streamlit app
CMD ["streamlit", "run", "medical_chatbot.py"]
