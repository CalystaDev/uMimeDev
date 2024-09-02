from fastapi import APIRouter, HTTPException, Request
from ..tasks import script_generation, voice_generation, image_generation, video_processing

router = APIRouter()

@router.get("/generate_script")
async def generate_script(request: Request):
    return script_generation(request)

@router.get("/generate_voice")
async def generate_voice(request: Request):
    return voice_generation(request)

@router.get("/generate_images")
async def generate_images(request: Request):
    return image_generation(request)

@router.get("/assemble_video")
async def assemble_video(request: Request):
    return video_processing(request)
