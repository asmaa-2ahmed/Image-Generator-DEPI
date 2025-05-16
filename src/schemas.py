from pydantic import BaseModel, Field
from typing import Optional

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    num_inference_steps: int = Field(default=30, ge=10, le=50)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=15.0)
    seed: Optional[int] = None
    save_dir: str = "src/assets/generated"

class GenerationResponse(BaseModel):
    filename: str
    filepath: str
