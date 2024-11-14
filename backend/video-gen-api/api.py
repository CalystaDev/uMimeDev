from fastapi import FastAPI, BackgroundTasks
from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips, CompositeVideoClip, AudioFileClip, TextClip
from moviepy.video.fx.all import resize
import requests
import base64
import json
import os
import datetime
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel
from google.cloud import storage
from prompts import prompts
import openai
import tempfile
import re
from typing import List, Tuple

storage_client = storage.Client()
BUCKET_NAME = 'background-vids'
VIDEO_FILE_NAME = 'subwaysurfers.mov'

# open_ai_api_key = 'KEY'
# eleven_labs_api_key = 'KEY'
open_ai_api_key = os.getenv("OPEN_AI_API_KEY")
eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")

app = FastAPI()

class VideoRequest(BaseModel):
    image_prompts: list
    script: str
    script_with_time_delimiter: str
    voice_id: str

class ScriptRequest(BaseModel):
    prompt: str
    voice_id: str

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
        title.strip()
        
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
        return image_prompts, title, script, script_with_dollars #returns image prompts, script, script with $ for image change
    except Exception as e:
        return [], f"An error occurred: {str(e)}", ""

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
    bottom_clip = VideoFileClip(video_path)  # Pass the file path instead of BytesIO
    bottom_clip = bottom_clip.resize(height=1920)
    bottom_clip = bottom_clip.crop(x_center=bottom_clip.w / 2, width=1080)
    width, height = bottom_clip.size
    bottom_half_clip = bottom_clip.crop(y1=height / 2 - 150, y2=height - 150)

    image_clips = []
    image_end_times = [words[i - 1][2] + 1 for i, word in enumerate(words) if word[0] == "$"]
    print(image_end_times)

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

def process_video_and_audio(script: str, script_with_time_delimiter: str, image_prompts: list, voice_id: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = download_video_from_gcp(temp_dir)  # This will download the video file
        # Generate images and audio concurrently
        with ThreadPoolExecutor() as executor:
            images_future = executor.submit(generate_images_from_script, image_prompts, temp_dir)
            audio_future = executor.submit(generate_audio, script, voice_id, temp_dir)
            images = images_future.result()
            print("NUM IMAGES: ", len(images))
            audio_file, response_dict = audio_future.result()

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

        final_video_path = create_video_with_audio(video_path, images, words, audio_file, temp_dir)
        upload_video_to_gcp(final_video_path, bucket_name="mimes", destination_blob_name=os.path.basename(final_video_path))

@app.post("/get-script-and-title")
async def get_script_and_title(request: ScriptRequest):
    prompt = construct_llm_prompt(request.prompt, request.voice_id)
    print("input to gpt:", prompt)
    image_prompts, title, script, script_with_times = generate_script_from_llm(prompt)
    print("audio script", script)
    print("script with times", script_with_times)
    print("title", title)
    return {"title": title, "script": script, "script_with_time_delimiter": script_with_times, "image_prompts": image_prompts}

@app.post("/generate-video")
async def generate_video(request: VideoRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(process_video_and_audio, request.script, request.script_with_time_delimiter, request.image_prompts, request.voice_id)
    return {"message": "Video is being processed", "status": "processing"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
