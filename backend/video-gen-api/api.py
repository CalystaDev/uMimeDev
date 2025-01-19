import os
import datetime
import json
import re
import requests
import base64
from typing import List, Tuple
import openai
from prompts import script_prompts, title_prompts
from google.cloud import storage
from moviepy.editor import VideoFileClip, TextClip, concatenate_videoclips, CompositeVideoClip, AudioFileClip
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import subprocess
import replicate

open_ai_api_key = os.getenv("OPEN_AI_API_KEY")
eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
BACKGROUND_BUCKET = 'backgroundvids'
MIMES_BUCKET = 'final-mimes'
MUSIC_BUCKET = 'backgroundmusicbucket'
storage_client = storage.Client()

generation_data = {}

def upload_to_gcs(local_file_path: str, bucket_name: str, destination_blob_name: str) -> str:
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
    os.remove(local_file_path) 
    return blob.public_url

def download_from_gcs(video_file_path: str, video_file_name: str, bucket_name: str):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(video_file_name)
    blob.download_to_filename(video_file_path)
    return video_file_path

def construct_llm_prompt(prompt: str, voice_id: str) -> str:
    base_prompt = script_prompts[voice_id]
    llm_prompt = base_prompt.replace("<promp-here>", prompt)
    return llm_prompt

def construct_title_llm_prompt(prompt: str, voice_id: str) -> str:
    base_prompt = title_prompts[voice_id]
    llm_prompt = base_prompt.replace("<promp-here>", prompt)
    return llm_prompt

def generate_script_from_llm(prompt: str) -> Tuple[List[str], str, str, str]:
    openai.api_key = open_ai_api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.8,
        )
        script_body = response.choices[0].message['content']
        script_lines = []
        script_with_dollars = []
        image_prompts = []
        lines = script_body.splitlines()
        for line in lines:
            if match := re.search(r'\[Image Prompt: (.*?)\]', line):
                image_prompts.append(match.group(1).strip())
                script_with_dollars.append("$")
            else:
                script_lines.append(line)
                script_with_dollars.append(line)
        script = "\n".join(script_lines).strip()
        script_with_dollars = "\n".join(script_with_dollars).strip()
        script = script.replace("—", "; ")
        script_with_dollars = script_with_dollars.replace("—", "- ")
        return image_prompts, script, script_with_dollars
    except Exception as e:
        return [], f"An error occurred: {str(e)}", "", ""

def generate_title_from_llm(prompt: str) -> str:
    try:
        url = "https://api.studio.nebius.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Authorization": f"Bearer {os.environ.get('NEBIUS_API_KEY')}"
        }
        payload = {
            "temperature": 0.8,
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        response = (requests.post(url, headers=headers, json=payload)).json()
        title = response['choices'][0]['message']['content']
        return title
    except Exception as e:
        return f"An error occurred with Nebius: {str(e)}"

# Lock to prevent race conditions during GCS uploads
gcs_lock = threading.Lock()

def generate_single_image(prompt: str, video_id: str, index: int) -> str:
    """
    Helper function to generate a single image and upload it to GCS.
    """
    try:
        output = replicate.run(
        "black-forest-labs/flux-schnell",
        input={"prompt": prompt}
        )
        local_image_path = f"image_{video_id}_{index}.png"
        with open(local_image_path, 'wb') as img_file:
            img_file.write(output[0].read())
        with gcs_lock:
            gcs_image_url = upload_to_gcs(local_image_path, MIMES_BUCKET, f"{video_id}/images/image_{index}.png")
        return gcs_image_url
    except Exception as e:
        print(f"Error generating image for prompt '{prompt}': {e}")
        return None

def generate_images_from_script(image_prompts: List[str], video_id: str) -> List[str]:
    """
    Generate images concurrently using ThreadPoolExecutor.
    """
    image_urls = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(generate_single_image, prompt, video_id, i) for i, prompt in enumerate(image_prompts)]
        for future in futures:
            result = future.result()
            if result:
                image_urls.append(result)
            else:
                image_urls.append(image_urls[-1])
    return image_urls


def generate_audio(script: str, script_with_time_delimiter: str, voice_id: str, video_id: str) -> str:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
    headers = {"Content-Type": "application/json", "xi-api-key": eleven_labs_api_key}
    data = {
        "text": script,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error encountered, status: {response.status_code}, content: {response.text}")
    response_dict = json.loads(response.content.decode("utf-8"))
    audio_bytes = base64.b64decode(response_dict["audio_base64"])
    audio_file_path = f"output_{video_id}.mp3"
    with open(audio_file_path, 'wb') as f:
        f.write(audio_bytes)
    # Upload audio to GCS
    gcs_audio_url = upload_to_gcs(audio_file_path, MIMES_BUCKET, f"{video_id}/audio/output.mp3")
    
    words = []
    curr_word = ''
    characters = response_dict['alignment']['characters']
    start_times = response_dict['alignment']['character_start_times_seconds']
    end_times = response_dict['alignment']['character_end_times_seconds']
    for i, char in enumerate(characters):
        if char == ' ':
            words.append((curr_word, start_times[i - len(curr_word)], end_times[i - 1]))
            curr_word = ''
        else:
            curr_word += char
    if curr_word:
        words.append((curr_word, start_times[len(start_times) - len(curr_word)], end_times[-1]))
    word_list = script_with_time_delimiter.split()
    word_count = 0
    print(f"Wordlist: {word_list}")
    print(f"Words 1: {words}")
    for word in word_list:
        if word == "$":
            if word_count > 0 and word_count <= len(words):
                previous_word = words[word_count - 1]
                words.insert(word_count, ("$", previous_word[1], previous_word[2]))
        else:
            word_count += 1
    print(f"Words 2: {words}")
    
    return gcs_audio_url, response_dict, words

def read_image_from_url(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image.shape[2] == 4:  # Check if the image has an alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image
        # return image
    else:
        raise ValueError(f"Failed to fetch image from URL: {url}, status code: {response.status_code}")

import cv2
import numpy as np

import cv2
import numpy as np

def apply_smooth_ken_burns(image, frame_idx, total_frames, start_scale=1.0, end_scale=1.2):
    """
    Apply a smooth Ken Burns effect (zoom-in) to an image with subpixel accuracy using OpenCV.

    Args:
        image: The input image (numpy array).
        frame_idx: The current frame index.
        total_frames: The total number of frames for the zoom effect.
        start_scale: Initial scale factor.
        end_scale: Final scale factor.

    Returns:
        The transformed image for the current frame.
    """
    # Calculate the current scale factor
    scale = start_scale + (end_scale - start_scale) * (frame_idx / total_frames)
    
    # Get the original dimensions
    h, w = image.shape[:2]
    
    # Calculate the new dimensions after scaling
    scaled_w = w / scale
    scaled_h = h / scale
    
    # Calculate translation offsets to center the zoom
    tx = (w - scaled_w) / 2
    ty = (h - scaled_h) / 2
    
    # Create the scaling and translation matrix
    scale_matrix = np.array([
        [scale, 0, -tx * scale],
        [0, scale, -ty * scale]
    ], dtype=np.float32)
    
    # Apply the transformation with subpixel accuracy
    smooth_image = cv2.warpAffine(image, scale_matrix, (w, h), flags=cv2.INTER_LINEAR)
    
    return smooth_image

def create_video_with_audio(video_path: str, image_urls: list, words: list, audio_url: str, video_id: str, background_music_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames in the background video
    video_duration = total_frames / fps
    output_file = f"final_video_with_audio_{video_id}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (255, 255, 255)  # White text
    outline_color = (0, 0, 0)     # Black outline
    thickness = 8                 # Thickness for the main text
    outline_thickness = 14         # Thickness for the outline

    background_frames = []

    print(f"Words: {words}")
    # Read all frames of the background video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        background_frames.append(frame)

    overlay_images = []
    print(f"Image URLs: image_urls")
    for i, image_url in enumerate(image_urls):
        print(f"Downloading image {i} from {image_url}")
        temp_path = download_from_gcs(f"/tmp/image_{i}.png", image_url.replace("https://storage.googleapis.com/final-mimes/", ""), 'final-mimes')
        img = cv2.imread(temp_path)
        if img is None:
            print(f"Error: Could not load image {temp_path}")
        overlay_images.append(img)

    print(f"Number of overlay images loaded: {len(overlay_images)}")

    image_end_times = [words[i - 1][2] + 1 for i, word in enumerate(words) if word[0] == "$"]
    
    print(f"Image End Times: {image_end_times}")
    total_video_duration = max(image_end_times[-1], video_duration)

    # Load the watermark
    watermark_path = download_from_gcs("/tmp/watermark.png", "watermark.png", "watermark-asset")
    if not watermark_path:
        raise RuntimeError("Error: Could not load watermark image.")

    watermark_cv2 = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
    if watermark_cv2 is None:
        raise RuntimeError("Error: Failed to read the watermark image with OpenCV.")

    # Resize the watermark
    watermark_height, watermark_width = watermark_cv2.shape[:2][0] // 16, watermark_cv2.shape[:2][1] // 16
    watermark_cv2 = cv2.resize(watermark_cv2, (watermark_width, watermark_height), interpolation=cv2.INTER_AREA)

    video_repeat_count = int(total_video_duration // video_duration) + 1  # Repeat the video as needed
    looped_background_frames = background_frames * video_repeat_count

    frame_idx = 0

    for repeat in range(video_repeat_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            bottom_half_frame = frame[height // 2 : height, 0: width]
            top_half_frame = np.zeros((height // 2, width, 3), dtype=np.uint8)
            
            image_found = False

            for i, img in enumerate(overlay_images):
                start = 0 if i == 0 else image_end_times[i - 1]
                end = image_end_times[i] if i != len(image_end_times) - 1 else total_video_duration
                # Adjust timing for doubled FPS
                if start <= frame_idx / (fps) <= end:
                    total_frames = int((end - start) * fps)
                    ken_burns_frame_idx = int((frame_idx / (fps) - start) * (fps))

                    img_resized = cv2.resize(img, (width, height // 2))
                    ken_burns_image = apply_smooth_ken_burns(img_resized, ken_burns_frame_idx, total_frames)

                    top_half_frame = frame[0:height // 2, 0:width]
                    top_half_frame = cv2.addWeighted(top_half_frame, 0, ken_burns_image, 1, 0)
                    frame[0:height // 2, 0:width] = top_half_frame
                    image_found = True
                    break

            if not image_found:
                if overlay_images:
                    last_img = overlay_images[-1]
                    img_resized = cv2.resize(last_img, (width, height // 2))
                    ken_burns_image = apply_smooth_ken_burns(img_resized, ken_burns_frame_idx, total_frames)

                    top_half_frame = frame[0:height // 2, 0:width]
                    top_half_frame = cv2.addWeighted(top_half_frame, 0, ken_burns_image, 1, 0)
                    frame[0:height // 2, 0:width] = top_half_frame

            for i, (text, start, end) in enumerate(words):
                if text == '$':
                    continue

                if start <= frame_idx / (fps) <= end:

                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    # Calculate the center position
                    center_x = (width - text_width) // 2  # Horizontal center
                    center_y = (height + text_height) // 2  # Vertical center
                    
                    text = text.replace("’", "'").replace("—", "-")

                    cv2.putText(frame, text, (center_x, center_y), font, font_scale, outline_color, outline_thickness, cv2.LINE_AA)
                    cv2.putText(frame, text, (center_x, center_y), font, font_scale, font_color, thickness, cv2.LINE_AA)           

            overlay = frame[20:20+watermark_height, 10:10+watermark_width]
            if watermark_cv2.shape[2] == 4:  # RGBA watermark
                watermark_rgb = watermark_cv2[:, :, :3]
                watermark_alpha = watermark_cv2[:, :, 3] / 255.0
                for c in range(3):
                    overlay[:, :, c] = (1 - watermark_alpha) * overlay[:, :, c] + watermark_alpha * watermark_rgb[:, :, c]
            else:
                overlay = cv2.addWeighted(overlay, 0, watermark_cv2, 1, 0)

            frame[20:20+watermark_height, 10:10+watermark_width] = overlay

            out.write(frame)
            frame_idx += 1

    cap.release()
    out.release()

    if not os.path.exists(output_file):
        raise RuntimeError(f"Error: Output video {output_file} was not created.")

    final_output = f"final_output_with_audio_{video_id}.mp4"
    
    audio_path = download_from_gcs(f"/tmp/audio.mp3", audio_url.replace("https://storage.googleapis.com/final-mimes/", ""), 'final-mimes')
    if not os.path.exists(audio_path):
        raise RuntimeError(f"Error: Audio file {audio_path} not found.")

    if background_music_path:
        # Use subprocess to run the ffmpeg command to re-encode video and combine with audio
        ffmpeg_command = [
            'ffmpeg',
            '-i', output_file,          # Input video file
            '-i', audio_path,           # Input primary audio file
            '-i', background_music_path,  # Input background music file
            '-filter_complex', (
                "[1:a]volume=2[a1];"
                "[2:a]volume=0.20[a2];"
                "[a1][a2]amix=inputs=2:duration=first:dropout_transition=2[a]"  # Mix the two audio streams
            ),
            '-map', '0:v',               # Map video from the video file
            '-map', '[a]',               # Map mixed audio
            '-c:v', 'libx264',           # Re-encode video with H.264 codec
            '-preset', 'fast',           # Use a faster encoding preset for efficiency
            '-crf', '23',                # Set quality for video
            '-c:a', 'aac',               # Re-encode audio with AAC codec
            '-b:a', '192k',              # Set audio bitrate for good quality
            '-movflags', '+faststart',   # Optimize for web streaming
            '-shortest',                 # Ensure the output duration matches the shortest input
            final_output                 # Output file
        ]
    else:
        # Use subprocess to run the ffmpeg command to re-encode video and combine with audio
        ffmpeg_command = [
            'ffmpeg',
            '-stream_loop', '-1',      # Loop the video indefinitely
            '-i', output_file,       # Input video file
            '-i', audio_path,        # Input audio file
            '-c:v', 'libx264',       # Re-encode video with H.264 codec
            '-preset', 'fast',       # Use a faster encoding preset for efficiency
            '-crf', '23',            # Set quality (lower is better; 23 is default for H.264)
            '-c:a', 'aac',           # Re-encode audio with AAC codec
            '-b:a', '192k',          # Set audio bitrate for good quality
            '-movflags', '+faststart',  # Optimize for web streaming
            '-shortest',             # Ensure the output duration matches the shortest input
            final_output             # Output file
        ]

    print(f"Running FFmpeg command: {' '.join(ffmpeg_command)}")

    # Execute FFmpeg command
    result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr.decode()}")
        raise RuntimeError("Error: FFmpeg command failed.")
    else:
        print("Video successfully re-encoded and audio combined.")

    # Upload the final video with audio to GCS
    gcs_video_url = upload_to_gcs(final_output, MIMES_BUCKET, f"{video_id}/final_video.mp4")
    return gcs_video_url


#initialize flask app
app = Flask(__name__)
CORS(app)

@app.route('/select_background/<video_id>', methods=['POST'])
def select_background(video_id):
    data = request.json
    background_id = data.get('background_id')
    if not background_id:
        return jsonify({"error": "Background ID is required"}), 400
    #@TODO: fix the ID-name mapping
    background_file_name = f"{background_id}.mov"
    try:
        video_path = download_from_gcs(f"/tmp/{background_file_name}", background_file_name, BACKGROUND_BUCKET)
        if video_id in generation_data:
            generation_data[video_id]['background_video_path'] = video_path
        else:
            generation_data[video_id] = {'background_video_path': video_path}
        return jsonify({"message": "Background video selected successfully", "video_path": video_path}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to select background video: {str(e)}"}), 500


@app.route('/select_background_music/<video_id>', methods=['POST'])
def select_background_music(video_id):
    data = request.json
    music_id = data.get('music_id')
    if not music_id:
        return jsonify({"error": "Music ID is required"}), 400
    if music_id == "none":
        if video_id in generation_data:
            generation_data[video_id]['background_music_path'] = None
        else:
            generation_data[video_id] = {'background_music_path': None}
        return jsonify({"message": "No background music selected"}), 200
    music_file_name = f"{music_id}.mp3"
    try:
        background_music_path = download_from_gcs(f"/tmp/{music_file_name}", music_file_name, MUSIC_BUCKET)
        if video_id in generation_data:
            generation_data[video_id]['background_music_path'] = background_music_path
        else:
            generation_data[video_id] = {'background_music_path': background_music_path}
        return jsonify({"message": "Background music selected successfully", "music_path": background_music_path}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to select background music: {str(e)}"}), 500

@app.route('/generate_title', methods=['POST'])
def generate_title():
    data = request.json
    prompt = data.get('prompt')
    voice_id = data.get('voice_id')
    if not prompt or not voice_id:
        return jsonify({"error": "Prompt and voice_id are required!!"}), 400
    llm_prompt = construct_title_llm_prompt(prompt, voice_id)
    title = generate_title_from_llm(llm_prompt)
    return jsonify({"title": title}), 200

@app.route('/generate_script', methods=['POST'])
def generate_script():
    data = request.json
    prompt = data.get('prompt')
    voice_id = data.get('voice_id')
    if not prompt or not voice_id:
        return jsonify({"error": "Prompt and voice_id are required"}), 400
    llm_prompt = construct_llm_prompt(prompt, voice_id)
    image_prompts, script, script_with_dollars = generate_script_from_llm(llm_prompt)
    if not script:
        return jsonify({"error": "Error generating script"}), 500
    video_id = str(datetime.datetime.now().timestamp())
    generation_data[video_id] = {
        'image_prompts': image_prompts,
        'script': script,
        'script_with_dollars': script_with_dollars
    }
    print(generation_data[video_id])
    return jsonify({"video_id": video_id, "script": script}), 200


@app.route('/generate_images/<video_id>', methods=['POST'])
def generate_images_endpoint(video_id):
    """
    HTTPS endpoint to generate images for a given video ID using optimized multithreading.
    """
    if video_id not in generation_data:
        return jsonify({"error": "Invalid video_id"}), 400

    # Retrieve data for the video ID
    data = generation_data[video_id]
    image_prompts = data['image_prompts']

    # Call the optimized function for generating images
    try:
        image_paths = generate_images_from_script(image_prompts, video_id)
        generation_data[video_id]['image_paths'] = image_paths
        return jsonify({"message": "Images generated successfully", "image_paths": image_paths}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to generate images: {str(e)}"}), 500


@app.route('/generate_audio/<video_id>', methods=['POST'])
def generate_audio_for_video(video_id):
    if video_id not in generation_data:
        return jsonify({"error": "Invalid video_id"}), 400

    voice_id = request.json.get('voice_id')
    data = generation_data[video_id]
    
    script = data['script'].replace("\n", " ").replace("—", "-").replace("#", "")
    script_with_dollars = data['script_with_dollars'].replace("\n", " ").replace("—", "-").replace("#", "")
    print(f"Script: {script}")
    print(f"Script w $: {script_with_dollars}")
    audio_file_path, _, words = generate_audio(script, script_with_dollars, voice_id, video_id)
    generation_data[video_id]['audio_file_path'] = audio_file_path
    generation_data[video_id]['words'] = words

    return jsonify({"message": "Audio generated", "audio_file": audio_file_path}), 200

@app.route('/create_video/<video_id>', methods=['POST'])
def create_video(video_id):
    if video_id not in generation_data:
        return jsonify({"error": "Invalid video_id"}), 400
    
    data = generation_data[video_id]
    video_path = data['background_video_path']
    image_paths = data['image_paths']
    words = data['words']
    audio_file_path = data['audio_file_path']
    background_music_path = generation_data[video_id].get('background_music_path')
    
    if not video_path or not image_paths or not words or not audio_file_path:
        return jsonify({"error": "Required data (background video, images, audio, or words) is missing"}), 400
    try:
        final_video_file = create_video_with_audio(video_path, image_paths, words, audio_file_path, video_id, background_music_path)
        return jsonify({"message": "Video created successfully", "video_file": final_video_file}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to create video: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
