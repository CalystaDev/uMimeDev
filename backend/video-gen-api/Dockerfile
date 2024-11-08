# Use an official Python image as a base
FROM python:3.10-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOST 0.0.0.0

# Install dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    imagemagick \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the custom policy.xml to overwrite the default ImageMagick policy
COPY policy.xml /etc/ImageMagick-6/policy.xml

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api.py /app
COPY prompts.py /app

# Start the FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
