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

# Modify ImageMagick policy.xml to allow necessary operations
RUN sed -i 's/<policy domain="path" rights="none" pattern="@\*" \/>/<policy domain="path" rights="read|write" pattern="@*" \/>/g' /etc/ImageMagick-6/policy.xml || true \
    && sed -i 's/<policy domain="coder" rights="none" pattern="PS" \/>/<policy domain="coder" rights="read|write" pattern="PS" \/>/g' /etc/ImageMagick-6/policy.xml || true

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
