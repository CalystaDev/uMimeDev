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

# Set up your API keys
open_ai_api_key = os.getenv("OPEN_AI_API_KEY")
eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
BUCKET_NAME = 'background-vids'
VIDEO_FILE_NAME = 'subwaysurfers.mov'

storage_client = storage.Client()

# In-memory storage for each part of the process
generation_data = {}

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

def generate_images_from_script(image_prompts: List[str], temp_dir: str) -> List[str]:
    image_paths = []
    openai.api_key = open_ai_api_key
    for i, prompt in enumerate(image_prompts):
        try:
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="512x512"
            )
            image_url = response['data'][0]['url']
            image_path = os.path.join(temp_dir, f"image_{i}.png")
            with open(image_path, 'wb') as img_file:
                img_file.write(requests.get(image_url).content)
            image_paths.append(image_path)
        except Exception as e:
            print(f"Error generating image for prompt '{prompt}': {e}")
    return image_paths

def generate_audio(script: str, voice_id: str, temp_dir: str) -> str:
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
    audio_filename = f"output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    audio_file_path = os.path.join(temp_dir, audio_filename)
    with open(audio_file_path, 'wb') as f:
        f.write(audio_bytes)
    return audio_file_path, response_dict

def create_video_with_audio(video_path: str, images: list, words: list, audio_file: str, temp_dir: str):
    bottom_clip = VideoFileClip(video_path)
    bottom_clip = bottom_clip.resize(height=1920)
    bottom_clip = bottom_clip.crop(x_center=bottom_clip.w / 2, width=1080)
    width, height = bottom_clip.size
    bottom_half_clip = bottom_clip.crop(y1=height / 2 - 150, y2=height - 150)

    image_clips = []
    image_end_times = [words[i - 1][2] + 1 for i, word in enumerate(words) if word[0] == "$"]

    for i, img in enumerate(images):
        img_clip = ImageClip(img).resize(height=1920/2).crop(x_center=bottom_clip.w/2, width=1080)
        img_clip = img_clip.set_start(0 if i == 0 else image_end_times[i - 1]).set_end(image_end_times[i] if i != len(image_end_times) - 1 else bottom_clip.duration)
        img_clip = img_clip.resize(width=width)
        ken_burns_clip = img_clip.fx(resize, lambda t: 1 + 0.02 * t)
        cropped_image_clip = ken_burns_clip.crop(y2=height / 2).set_position(("center", "top"))
        image_clips.append(cropped_image_clip)

    top_half_clip = concatenate_videoclips(image_clips, method="compose", padding=-0.2)
    final_clip = CompositeVideoClip([top_half_clip, bottom_half_clip.set_position(("center", "bottom"))], size=(width, height))
    audio_clip = AudioFileClip(audio_file)
    final_clip = final_clip.set_duration(audio_clip.duration)
    final_clip = final_clip.set_audio(audio_clip)
    caption_clips = []
    for text, start, end in words:
        if text == '$':
            continue
        word_duration = (end - start)
        word_clip = TextClip(text, fontsize=100, color='white', stroke_color='black', stroke_width=2, font='Impact', align='center')
        word_clip = word_clip.set_position(('center', 'center')).set_start(start).set_duration(word_duration).crossfadein(0.1)
        caption_clips.append(word_clip)
    final_video_with_captions = CompositeVideoClip([final_clip] + caption_clips)
    output_file = os.path.join(temp_dir, f"final_video_with_audio_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    final_video_with_captions.write_videofile(output_file, audio_codec="aac")
    return output_file

def download_video_from_gcp(temp_dir: str):
    video_file_path = os.path.join(temp_dir, VIDEO_FILE_NAME)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(VIDEO_FILE_NAME)
    blob.download_to_filename(video_file_path)
    return video_file_path

def upload_video_to_gcp(file_path: str, bucket_name: str, destination_blob_name: str):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    print(f"File {file_path} uploaded to {bucket_name}/{destination_blob_name}.")

# Initialize Flask app
app = Flask(__name__)

@app.route('/generate_script', methods=['POST'])
def generate_script():
    # Get data from the request
    data = request.json
    prompt = data.get('prompt')
    voice_id = data.get('voice_id')
    
    if not prompt or not voice_id:
        return jsonify({"error": "Prompt and voice_id are required"}), 400

    llm_prompt = construct_llm_prompt(prompt, voice_id)
    
    # Generate script from LLM
    image_prompts, title, script, script_with_dollars = generate_script_from_llm(llm_prompt)
    
    if not script:
        return jsonify({"error": "Error generating script"}), 500
    
    # Save data in the in-memory storage
    video_id = str(datetime.datetime.now().timestamp())  # Unique ID for this video generation session
    generation_data[video_id] = {
        'image_prompts': image_prompts,
        'title': title,
        'script': script,
        'script_with_dollars': script_with_dollars
    }
    
    return jsonify({"video_id": video_id, "title": title, "script": script}), 200

@app.route('/generate_images/<video_id>', methods=['POST'])
def generate_images(video_id):
    # Check if video_id exists in generation data
    if video_id not in generation_data:
        return jsonify({"error": "Invalid video_id"}), 400

    data = generation_data[video_id]
    image_prompts = data['image_prompts']
    
    # Create temporary directory for files
    temp_dir = os.path.join("/tmp", str(datetime.datetime.now().timestamp()))
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate images
    image_paths = generate_images_from_script(image_prompts, temp_dir)
    
    # Save image paths in generation data
    generation_data[video_id]['image_paths'] = image_paths

    return jsonify({"message": "Images generated", "image_paths": image_paths}), 200

@app.route('/generate_audio/<video_id>', methods=['POST'])
def generate_audio_for_video(video_id):
    # Check if video_id exists in generation data
    if video_id not in generation_data:
        return jsonify({"error": "Invalid video_id"}), 400

    data = generation_data[video_id]
    script = data['script_with_dollars']
    
    # Create temporary directory for files
    temp_dir = os.path.join("/tmp", str(datetime.datetime.now().timestamp()))
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate audio
    voice_id = "voice_1"  # Assuming this is passed from the client or set earlier
    audio_file_path, _ = generate_audio(script, voice_id, temp_dir)
    
    # Save audio path in generation data
    generation_data[video_id]['audio_file_path'] = audio_file_path

    return jsonify({"message": "Audio generated", "audio_file": audio_file_path}), 200

@app.route('/create_video/<video_id>', methods=['POST'])
def create_video(video_id):
    # Check if video_id exists in generation data
    if video_id not in generation_data:
        return jsonify({"error": "Invalid video_id"}), 400
    
    data = generation_data[video_id]
    image_paths = data['image_paths']
    script = data['script']
    audio_file_path = data['audio_file_path']
    
    # Assuming the video path is predefined or uploaded
    video_path = "sample_video.mp4"  # Example placeholder

    # Create video
    final_video_file = create_video_with_audio(video_path, image_paths, script.split("\n"), audio_file_path, "/tmp")
    
    # Upload to GCP
    upload_video_to_gcp(final_video_file, BUCKET_NAME, VIDEO_FILE_NAME)
    
    return jsonify({"message": "Video created successfully", "video_file": final_video_file}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
