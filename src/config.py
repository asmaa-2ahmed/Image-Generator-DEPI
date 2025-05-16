import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES_DIR = os.path.join(BASE_DIR, "assets", "files")
GENERATED_DIR = os.path.join(BASE_DIR, "assets", "generated")

# Strictly get values from the environment without defaults
MODEL_NAME = os.getenv("MODEL_NAME")
VAE_NAME = os.getenv("VAE_NAME")
LORA_PATH = os.getenv("LORA_PATH")

# Optional: raise an error if any are missing
required_vars = {
    "MODEL_NAME": MODEL_NAME,
    "VAE_NAME": VAE_NAME,
    "LORA_PATH": LORA_PATH,
}

missing_vars = [k for k, v in required_vars.items() if not v]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")
