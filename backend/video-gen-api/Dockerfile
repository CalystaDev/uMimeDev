# Use a lightweight Python base image
FROM python:3.10-slim

# Set environment variables to prevent .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for video and image processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    imagemagick \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Update ImageMagick policy to allow certain image operations
COPY policy.xml /etc/ImageMagick-6/policy.xml

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the Docker image
COPY api.py /app
COPY prompts.py /app

# Expose the port the app will run on
EXPOSE 5000

# Run the application with Flask
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
