FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    ffmpeg \
    imagemagick \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY policy.xml /etc/ImageMagick-6/policy.xml

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py /app
COPY prompts.py /app

ENV FLASK_APP=api.py
ENV PORT=8080
EXPOSE 8080

CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
