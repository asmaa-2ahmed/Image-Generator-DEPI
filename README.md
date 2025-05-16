# Image Generator DEPI - Naruto Edition

A deep learning project that fine-tunes Stable Diffusion XL (SDXL) for generating anime-style images inspired by the Naruto series.

## Overview

This project uses the DreamBooth LoRA fine-tuning technique on Stable Diffusion XL to create a specialized image generator for Naruto-style anime characters. The model is trained on the lambdalabs/naruto-blip-captions dataset from Hugging Face.

## Features

- Fine-tunes SDXL using LoRA adapters for efficient training
- Optimized inference with VAE-fp16 fixes
- HuggingFace integration for model hosting and dataset access
- Memory management optimizations for inference
- Interactive visualization of the training dataset

## Project Structure

- `notebook.ipynb`: Complete Jupyter notebook with:
  - Dataset exploration and visualization
  - Model training configuration
  - SDXL LoRA fine-tuning process
  - Memory management utilities
  - Inference pipeline setup and testing
- `train_dreambooth_lora_sdxl_advanced.py`: Training script downloaded during notebook execution

## Setup Instructions

### Requirements

To run this project, you need:

1. Python 3.8+ environment
2. GPU with CUDA support (minimum 16GB VRAM recommended)
3. Hugging Face account with API token
4. Kaggle account with API token (for accessing secrets)

The notebook will install all required dependencies automatically:
```
datasets, huggingface_hub, xformers, bitsandbytes, transformers, accelerate, 
peft, dadaptation, prodigyopt, torchvision, python-slugify, diffusers
```

### Environment Setup

The notebook configures accelerate automatically. No manual setup is needed beyond providing your Hugging Face token.

## Usage Guide

The notebook provides an end-to-end workflow:

1. **Install Dependencies**: The first cell installs all required packages and downloads the training script.

2. **Data Exploration**: Visualize sample images from the Naruto dataset with their captions.

3. **Training Preparation**: Configure accelerate, log in to Hugging Face, and set parameters like model name and LoRA rank.

4. **Model Training**: Fine-tune SDXL with LoRA on the Naruto dataset (takes several hours on a good GPU).

5. **Memory Management**: Utilities to clear CUDA memory between pipeline changes.

6. **Inference**: Load the trained model and generate custom Naruto-style images from text prompts.

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
