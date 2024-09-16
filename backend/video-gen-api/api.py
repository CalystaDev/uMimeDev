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

app = FastAPI()

# Paths for testing
dummy_images = [
    "/Users/mohit/Downloads/uMime/uMimeDev/backend/video-gen-api/dummy_images/image1.jpeg",
    "/Users/mohit/Downloads/uMime/uMimeDev/backend/video-gen-api/dummy_images/image2.jpeg",
    "/Users/mohit/Downloads/uMime/uMimeDev/backend/video-gen-api/dummy_images/image3.jpeg",
    "/Users/mohit/Downloads/uMime/uMimeDev/backend/video-gen-api/dummy_images/image4.jpeg",
    "/Users/mohit/Downloads/uMime/uMimeDev/backend/video-gen-api/dummy_images/image5.jpeg"
]
video_path = "/Users/mohit/Downloads/uMime/uMimeDev/backend/video-gen-api/dummy_images/subwaysurfers.mov"

def generate_audio(script: str, voice_id: str, api_key: str) -> str:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
    headers = {"Content-Type": "application/json", "xi-api-key": api_key}

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
    audio_file_path = os.path.join(os.getcwd(), audio_filename)

    with open(audio_file_path, 'wb') as f:
        f.write(audio_bytes)

    return audio_file_path, response_dict

def create_video_with_audio(video: str, images: list, words: list, audio_file: str):
    bottom_clip = VideoFileClip(video)
    bottom_clip = bottom_clip.resize(height=1920)
    bottom_clip = bottom_clip.crop(x_center=bottom_clip.w / 2, width=1080)
    width, height = bottom_clip.size
    bottom_half_clip = bottom_clip.crop(y1=height / 2 - 150, y2=height - 150)

    image_clips = []
    image_end_times = [words[i][2] + 1 for i, word in enumerate(words) if word[0] == "$"]

    for i, img in enumerate(images):
        img_clip = ImageClip(img).resize(height=1920/2).crop(x_center=bottom_clip.w/2, width=1080)
        img_clip = img_clip.set_start(0 if i == 0 else image_end_times[i - 1]).set_end(image_end_times[i])
        ken_burns_clip = img_clip.fx(resize, lambda t: 1 + 0.02 * t)
        cropped_image_clip = ken_burns_clip.crop(y2=height / 2).set_position(("center", "top"))
        image_clips.append(cropped_image_clip)

    top_half_clip = concatenate_videoclips(image_clips, method="compose", padding=-0.2)

    final_clip = CompositeVideoClip([top_half_clip, bottom_half_clip.set_position(("center", "bottom"))], size=(width, height))

    audio_clip = AudioFileClip(audio_file)

    final_clip = final_clip.set_audio(audio_clip)
    caption_clips = []
    for text, start, end in words:
        if text == '$':
            continue
        word_duration = (end - start)
        word_clip = TextClip(text, fontsize=100, color='white', stroke_color='black', stroke_width=2, size=(width / 3, 500), font='Impact', align='center')
        word_clip = word_clip.set_position(('center', 'center')).set_start(start).set_duration(word_duration).crossfadein(0.1)
        caption_clips.append(word_clip)

    final_video_with_captions = CompositeVideoClip([final_clip] + caption_clips)
    output_file = f"final_video_with_audio_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    final_video_with_captions.write_videofile(output_file, audio_codec="aac")

    return output_file

def process_video_and_audio(script: str, voice_id: str, api_key: str, video: str, images: list):
    with ThreadPoolExecutor() as executor:
        audio_future = executor.submit(generate_audio, script, voice_id, api_key)
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

        final_video = create_video_with_audio(video, images, words, audio_file)

    return final_video

class VideoRequest(BaseModel):
    prompt: str
    voice_id: str
    api_key: str

@app.post("/generate-video")
async def generate_video(request: VideoRequest, background_tasks: BackgroundTasks):
    script = request.prompt
    voice_id = request.voice_id
    api_key = request.api_key

    background_tasks.add_task(process_video_and_audio, script, voice_id, api_key, video_path, dummy_images)
    return {"message": "Video is being processed", "status": "processing"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



