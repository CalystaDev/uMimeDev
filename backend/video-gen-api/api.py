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

def upload_to_gcs(local_file_path: str, bucket_name: str, destination_blob_name: str) -> str:
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
    os.remove(local_file_path)  # Optional: Remove the local file after upload
    return blob.public_url  # Return the public URL of the file

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

def generate_images_from_script(image_prompts: List[str], video_id: str) -> List[str]:
    image_urls = []
    openai.api_key = open_ai_api_key
    for i, prompt in enumerate(image_prompts):
        try:
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="512x512"
            )
            image_url = response['data'][0]['url']
            local_image_path = f"image_{video_id}_{i}.png"
            with open(local_image_path, 'wb') as img_file:
                img_file.write(requests.get(image_url).content)
            # Upload image to GCS
            gcs_image_url = upload_to_gcs(local_image_path, BUCKET_NAME, f"{video_id}/images/image_{i}.png")
            image_urls.append(gcs_image_url)
        except Exception as e:
            print(f"Error generating image for prompt '{prompt}': {e}")
    return image_urls

def generate_audio(script: str, voice_id: str, video_id: str) -> str:
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
    gcs_audio_url = upload_to_gcs(audio_file_path, BUCKET_NAME, f"{video_id}/audio/output.mp3")
    return gcs_audio_url, response_dict

def create_video_with_audio(video_path: str, image_urls: list, words: list, audio_url: str, video_id: str):
    bottom_clip = VideoFileClip(video_path)
    bottom_clip = bottom_clip.resize(height=1920)
    bottom_clip = bottom_clip.crop(x_center=bottom_clip.w / 2, width=1080)
    width, height = bottom_clip.size
    bottom_half_clip = bottom_clip.crop(y1=height / 2 - 150, y2=height - 150)

    image_clips = []
    image_end_times = [words[i - 1][2] + 1 for i, word in enumerate(words) if word[0] == "$"]

    for i, image_url in enumerate(image_urls):
        img_clip = ImageClip(image_url).resize(height=1920/2).crop(x_center=bottom_clip.w/2, width=1080)
        img_clip = img_clip.set_start(0 if i == 0 else image_end_times[i - 1]).set_end(image_end_times[i] if i != len(image_end_times) - 1 else bottom_clip.duration)
        img_clip = img_clip.resize(width=width)
        ken_burns_clip = img_clip.fx(resize, lambda t: 1 + 0.02 * t)
        cropped_image_clip = ken_burns_clip.crop(y2=height / 2).set_position(("center", "top"))
        image_clips.append(cropped_image_clip)

    top_half_clip = concatenate_videoclips(image_clips, method="compose", padding=-0.2)
    final_clip = CompositeVideoClip([top_half_clip, bottom_half_clip.set_position(("center", "bottom"))], size=(width, height))
    audio_clip = AudioFileClip(audio_url)
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
    output_file = f"final_video_with_audio_{video_id}.mp4"
    final_video_with_captions.write_videofile(output_file, audio_codec="aac")
    # Upload final video to GCS
    gcs_video_url = upload_to_gcs(output_file, BUCKET_NAME, f"{video_id}/final_video.mp4")
    return gcs_video_url
