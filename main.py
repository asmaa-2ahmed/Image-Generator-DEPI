from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from src.schemas import GenerationRequest, GenerationResponse
from src.inference import generate_image
import uuid
import os

app = FastAPI(title="Naruto Image Generator API")

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    image = generate_image(request)
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(request.save_dir, filename)
    image.save(filepath)
    return GenerationResponse(filename=filename, filepath=filepath)

@app.get("/image/{filename}")
async def get_image(filename: str):
    path = os.path.join("src/assets/generated", filename)
    return FileResponse(path)
