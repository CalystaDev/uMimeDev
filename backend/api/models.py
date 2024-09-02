from pydantic import BaseModel

class ScriptRequest(BaseModel):
    prompt: str

class VoiceRequest(BaseModel):
    script: str

class ImageRequest(BaseModel):
    prompt: str

class VideoRequest(BaseModel):
    images: list
    audio: str
    video_background: str #IDK WHAT TO DO HERE

class ScriptResponse(BaseModel):
    script: str

class VideoResponse(BaseModel):
    video_file: str
