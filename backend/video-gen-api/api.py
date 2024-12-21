import os
import datetime
import json
import re
import requests
import base64
from typing import List, Tuple
import openai
from prompts import prompts
from google.cloud import storage
from moviepy.editor import VideoFileClip, TextClip, concatenate_videoclips, CompositeVideoClip, AudioFileClip
from flask import Flask, request, jsonify
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

open_ai_api_key = os.getenv("OPEN_AI_API_KEY")
eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
BACKGROUND_BUCKET = 'background-vids'
MIMES_BUCKET = 'mimes'
VIDEO_FILE_NAME = 'subwaysurfers.mov'

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
    base_prompt = prompts[voice_id]
    llm_prompt = base_prompt.replace("<promp-here>", prompt)
    return llm_prompt

def generate_script_from_llm(prompt: str) -> Tuple[List[str], str, str, str]:
    openai.api_key = open_ai_api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7,
        )
        response_content = response.choices[0].message['content']
        title, script_body = response_content.split("#####", 1)
        title = title.strip()
        
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
        return image_prompts, title, script, script_with_dollars
    except Exception as e:
        return [], f"An error occurred: {str(e)}", "", ""

# def generate_images_from_script(image_prompts: List[str], video_id: str) -> List[str]:
#     image_urls = []
#     openai.api_key = open_ai_api_key
#     for i, prompt in enumerate(image_prompts):
#         try:
#             response = openai.Image.create(
#                 prompt=prompt,
#                 n=1,
#                 size="512x512"
#             )
#             image_url = response['data'][0]['url']
#             local_image_path = f"image_{video_id}_{i}.png"
#             with open(local_image_path, 'wb') as img_file:
#                 img_file.write(requests.get(image_url).content)
#             gcs_image_url = upload_to_gcs(local_image_path, BUCKET_NAME, f"{video_id}/images/image_{i}.png")
#             image_urls.append(gcs_image_url)
#         except Exception as e:
#             print(f"Error generating image for prompt '{prompt}': {e}")
#     return image_urls


# Lock to prevent race conditions during GCS uploads
gcs_lock = threading.Lock()

def generate_single_image(prompt: str, video_id: str, index: int) -> str:
    """
    Helper function to generate a single image and upload it to GCS.
    """
    openai.api_key = open_ai_api_key
    try:
        response = openai.Image.create(prompt=prompt, n=1, size="512x512")
        image_url = response['data'][0]['url']
        
        local_image_path = f"image_{video_id}_{index}.png"
        with open(local_image_path, 'wb') as img_file:
            img_file.write(requests.get(image_url).content)
        
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
    word_list = script_with_time_delimiter.split()
    word_count = 0
    for word in word_list:
        if word == "$":
            if word_count > 0 and word_count <= len(words):
                previous_word = words[word_count - 1]
                words.insert(word_count, ("$", previous_word[1], previous_word[2]))
        else:
            word_count += 1
    return gcs_audio_url, response_dict, words

# def create_video_with_audio(video_path: str, image_urls: list, words: list, audio_url: str, video_id: str):
#     bottom_clip = VideoFileClip(video_path)
#     bottom_clip = bottom_clip.resize(height=1920)
#     bottom_clip = bottom_clip.crop(x_center=bottom_clip.w / 2, width=1080)
#     width, height = bottom_clip.size
#     bottom_half_clip = bottom_clip.crop(y1=height / 2 - 150, y2=height - 150)

#     image_clips = []
#     image_end_times = [words[i - 1][2] + 1 for i, word in enumerate(words) if word[0] == "$"]

#     for i, image_url in enumerate(image_urls):
#         img_clip = ImageClip(image_url).resize(height=1920/2).crop(x_center=bottom_clip.w/2, width=1080)
#         img_clip = img_clip.set_start(0 if i == 0 else image_end_times[i - 1]).set_end(image_end_times[i] if i != len(image_end_times) - 1 else bottom_clip.duration)
#         img_clip = img_clip.resize(width=width)
#         ken_burns_clip = img_clip.fx(resize, lambda t: 1 + 0.02 * t)
#         cropped_image_clip = ken_burns_clip.crop(y2=height / 2).set_position(("center", "top"))
#         image_clips.append(cropped_image_clip)

#     top_half_clip = concatenate_videoclips(image_clips, method="compose", padding=-0.2)
#     final_clip = CompositeVideoClip([top_half_clip, bottom_half_clip.set_position(("center", "bottom"))], size=(width, height))
#     audio_clip = AudioFileClip(audio_url)
#     final_clip = final_clip.set_duration(audio_clip.duration)
#     final_clip = final_clip.set_audio(audio_clip)
#     caption_clips = []
#     for text, start, end in words:
#         if text == '$':
#             continue
#         word_duration = (end - start)
#         word_clip = TextClip(text, fontsize=100, color='white', stroke_color='black', stroke_width=2, font='Impact', align='center')
#         word_clip = word_clip.set_position(('center', 'center')).set_start(start).set_duration(word_duration).crossfadein(0.1)
#         caption_clips.append(word_clip)
#     final_video_with_captions = CompositeVideoClip([final_clip] + caption_clips)
#     output_file = f"final_video_with_audio_{video_id}.mp4"
#     final_video_with_captions.write_videofile(output_file, audio_codec="aac")
#     # Upload final video to GCS
#     gcs_video_url = upload_to_gcs(output_file, BUCKET_NAME, f"{video_id}/final_video.mp4")
#     return gcs_video_url

def read_image_from_url(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return image
    else:
        raise ValueError(f"Failed to fetch image from URL: {url}, status code: {response.status_code}")

def create_video_with_audio(video_path: str, image_urls: list, words: list, audio_url: str, video_id: str):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_file = f"final_video_with_audio_{video_id}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    overlay_images = []
    print(f"Image URLs: image_urls")
    for i, image_url in enumerate(image_urls):
        print(f"Downloading image {i} from {image_url}")
        temp_path = download_from_gcs(f"/tmp/image_{i}.png", image_url.replace("https://storage.googleapis.com/mimes/", ""), 'mimes')
        img = cv2.imread(temp_path)
        if img is None:
            print(f"Error: Could not load image {temp_path}")
        overlay_images.append(img)

    print(f"Number of overlay images loaded: {len(overlay_images)}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (255, 255, 255)
    thickness = 2

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"End of video reached at frame {frame_idx}")
            break
        
        if frame is None:
            print(f"Warning: Empty frame at {frame_idx}")
            continue

        for i, (text, start, end) in enumerate(words):
            if text == '$':
                continue
            if start <= frame_idx / fps <= end:
                print(f"Adding text '{text}' at frame {frame_idx}")
                cv2.putText(frame, text, (50, 50), font, font_scale, font_color, thickness, cv2.LINE_AA)
                if i < len(overlay_images):
                    img = cv2.resize(overlay_images[i], (width, height))
                    alpha = 0.6
                    frame = cv2.addWeighted(frame, 1 - alpha, img, alpha, 0)
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()

    if not os.path.exists(output_file):
        raise RuntimeError(f"Error: Output video {output_file} was not created.")

    final_output = f"final_output_with_audio_{video_id}.mp4"

    audio_path = download_from_gcs(f"/tmp/audio.mp3", audio_url.replace("https://storage.googleapis.com/mimes/", ""), 'mimes')
    ffmpeg_command = f'ffmpeg -i {output_file} -i {audio_path} -c:v copy -c:a aac {final_output}'
    print(f"Running FFmpeg command: {ffmpeg_command}")
    result = os.system(ffmpeg_command)

    if result != 0:
        raise RuntimeError("Error: FFmpeg command failed.")

    gcs_video_url = upload_to_gcs(output_file, MIMES_BUCKET, f"{video_id}/final_video.mp4")
    return gcs_video_url


#initialize flask app
app = Flask(__name__)

@app.route('/select_background/<video_id>', methods=['POST'])
def select_background(video_id):
    data = request.json
    background_id = data.get('background_id')
    if not background_id:
        return jsonify({"error": "Background ID is required"}), 400
    #@TODO: fix the ID-name mapping
    background_file_name = f"{background_id}.mp4"
    try:
        video_path = download_from_gcs(f"/tmp/{background_file_name}", background_file_name, BACKGROUND_BUCKET)
        if video_id in generation_data:
            generation_data[video_id]['background_video_path'] = video_path
        else:
            generation_data[video_id] = {'background_video_path': video_path}
        return jsonify({"message": "Background video selected successfully", "video_path": video_path}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to select background video: {str(e)}"}), 500

@app.route('/generate_script', methods=['POST'])
def generate_script():
    data = request.json
    prompt = data.get('prompt')
    voice_id = data.get('voice_id')
    
    if not prompt or not voice_id:
        return jsonify({"error": "Prompt and voice_id are required"}), 400

    llm_prompt = construct_llm_prompt(prompt, voice_id)
    image_prompts, title, script, script_with_dollars = generate_script_from_llm(llm_prompt)
    
    if not script:
        return jsonify({"error": "Error generating script"}), 500
    
    video_id = str(datetime.datetime.now().timestamp())
    generation_data[video_id] = {
        'image_prompts': image_prompts,
        'title': title,
        'script': script,
        'script_with_dollars': script_with_dollars
    }
    
    print(generation_data[video_id])

    return jsonify({"video_id": video_id, "title": title, "script": script}), 200

# @app.route('/generate_images/<video_id>', methods=['POST'])
# def generate_images(video_id):
#     if video_id not in generation_data:
#         return jsonify({"error": "Invalid video_id"}), 400

#     data = generation_data[video_id]
#     image_prompts = data['image_prompts']
#     image_paths = generate_images_from_script(image_prompts, video_id)
    
#     generation_data[video_id]['image_paths'] = image_paths
#     return jsonify({"message": "Images generated", "image_paths": image_paths}), 200

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

    data = generation_data[video_id]
    script = data['script']
    script_with_dollars = data['script_with_dollars']
    audio_file_path, _, words = generate_audio(script, script_with_dollars, "jsCqWAovK2LkecY7zXl4", video_id)
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
    if not video_path or not image_paths or not words or not audio_file_path:
        return jsonify({"error": "Required data (background video, images, audio, or words) is missing"}), 400
    try:
        final_video_file = create_video_with_audio(video_path, image_paths, words, audio_file_path, video_id)
        return jsonify({"message": "Video created successfully", "video_file": final_video_file}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to create video: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
