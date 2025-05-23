# Image Generator DEPI - Naruto Edition

A deep learning project that fine-tunes Stable Diffusion XL (SDXL) for generating anime-style images inspired by the Naruto series.

## Overview

This project uses the DreamBooth LoRA fine-tuning technique on Stable Diffusion XL to create a specialized image generator for Naruto-style anime characters. The model is trained on the lambdalabs/naruto-blip-captions dataset from Hugging Face.

## Demo

https://huggingface.co/spaces/AbdelrahmanGalhom/Naruto-Diffuser-FineTuned

![Demo](DemoGif.gif)

You can try the model directly using our Hugging Face Space linked above, or check out the trained model at:
https://huggingface.co/AbdelrahmanGalhom/diffusers-finetuned-naruto

## Features

- Fine-tunes SDXL using LoRA adapters for efficient training
- Optimized inference with VAE-fp16 fixes
- HuggingFace integration for model hosting and dataset access
- Memory management optimizations for inference
- Interactive visualization of the training dataset
- Deployed demo on Hugging Face Spaces
- FastAPI implementation for serving images via REST API

## Project Structure

- `docs/` 
  - `Presentation.pptx`: Project presentation slides
  - `Report.pdf`: Detailed project report
- `src/`
  - `assets/`: Contains image resources
  - `config.py`: Configuration settings for the model
  - `inference.py`: Image generation logic
  - `schemas.py`: Data models for API requests/responses
- `.env.example`: Example environment variables file
- `.gitignore`: Git ignore rules
- `Demo.mp4`: Video demonstration of the project
- `DemoGif.gif`: Animated demo of the application
- `main.py`: FastAPI application entry point
- `notebook.ipynb`: Jupyter notebook with training code
- `README.md`: Project documentation
- `requirements.txt`: Project dependencies

## Setup Instructions

### Requirements

To run this project, you need:

1. Python 3.8+ environment
2. GPU with CUDA support (minimum 16GB VRAM recommended)
3. Hugging Face account with API token
4. Kaggle account with API token (for accessing secrets)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Image-Generator-DEPI.git
cd Image-Generator-DEPI
```

2. Install dependencies using the requirements.txt file:
```bash
pip install -r requirements.txt
```

The requirements.txt includes all necessary dependencies:
```
datasets
huggingface_hub
xformers
bitsandbytes
transformers
accelerate
peft
dadaptation
prodigyopt
torchvision
python-slugify
diffusers
fastapi
uvicorn
pillow
python-dotenv
```

## API Usage

The project includes a FastAPI application that serves generated images:

### Endpoints

- `POST /generate`: Generate a new Naruto-style image from a text prompt
  - Request body: JSON with prompt and configuration
  - Returns: Image filename and path

- `GET /image/{filename}`: Retrieve a previously generated image
  - Path parameter: Image filename
  - Returns: The image file

### Running the API

```bash
uvicorn main:app --reload
```

Visit http://localhost:8000/docs to access the interactive API documentation.

## Usage Guide

The notebook provides an end-to-end workflow:

1. **Install Dependencies**: The first cell installs all required packages and downloads the training script.

2. **Data Exploration**: Visualize sample images from the Naruto dataset with their captions.

3. **Training Preparation**: Configure accelerate, log in to Hugging Face, and set parameters like model name and LoRA rank.

4. **Model Training**: Fine-tune SDXL with LoRA on the Naruto dataset (takes several hours on a good GPU).

5. **Memory Management**: Utilities to clear CUDA memory between pipeline changes.

6. **Inference**: Load the trained model and generate custom Naruto-style images from text prompts.

## Using the Deployed Model

You can easily generate Naruto-style images using:

1. Our Hugging Face Space: [Naruto-Diffuser-FineTuned](https://huggingface.co/spaces/AbdelrahmanGalhom/Naruto-Diffuser-FineTuned)
2. Directly with the model: [diffusers-finetuned-naruto](https://huggingface.co/AbdelrahmanGalhom/diffusers-finetuned-naruto)

## Example Prompts

Try these prompts with your trained model:
- "a naruto anime character with red hair and green eyes"
- "a naruto anime character in action pose with blue chakra"
- "a naruto anime character with headband and whiskered face"

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for hosting models and datasets
- [Lambda Labs](https://lambdalabs.com/) for the Naruto dataset
- [Stability AI](https://stability.ai/) for Stable Diffusion XL
- [Diffusers library](https://github.com/huggingface/diffusers) for the training scripts and pipeline

## License

This project uses models and code that may have their own licenses. Please refer to:
- [Stable Diffusion XL License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [Diffusers License](https://github.com/huggingface/diffusers/blob/main/LICENSE)
