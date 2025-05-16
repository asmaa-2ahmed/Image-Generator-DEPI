import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from src.config import MODEL_NAME, VAE_NAME, LORA_PATH
from src.schemas import GenerationRequest

model = None

def load_model():
    global model
    if model is not None:
        return model

    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )

    pipe.vae = pipe.vae.from_pretrained(
        VAE_NAME,
        torch_dtype=torch.float16
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True
    )

    if LORA_PATH:
        pipe.load_lora_weights(LORA_PATH)

    pipe.enable_xformers_memory_efficient_attention()

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    model = pipe
    return model

def generate_image(request: GenerationRequest):
    model = load_model()
    generator = torch.Generator().manual_seed(request.seed) if request.seed else None

    if "naruto" in request.prompt.lower() and "anime" not in request.prompt.lower():
        prompt = f"a naruto anime character, {request.prompt}"
    else:
        prompt = request.prompt

    image = model(
        prompt=prompt,
        negative_prompt=request.negative_prompt,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        generator=generator,
    ).images[0]

    return image
