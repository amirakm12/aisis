import os
import torch
from bark import preload_models
import whisper
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
from urllib.request import urlretrieve

# Directory for models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Bark
preload_models()

# Whisper
whisper.load_model("small", download_root=os.path.join(MODEL_DIR, "whisper/small"))  # Using small model as per query for multilingual
whisper.load_model("small.en", download_root=os.path.join(MODEL_DIR, "whisper/small.en"))

# Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.save_pretrained(os.path.join(MODEL_DIR, "diffusion/sd-v1-4"))

# Phi-2
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model.save_pretrained(os.path.join(MODEL_DIR, "llm/phi-2"))
tokenizer.save_pretrained(os.path.join(MODEL_DIR, "llm/phi-2"))

# Llama-2-7b-chat GGUF for llama-cpp
llama_dir = os.path.join(MODEL_DIR, "llm/llama-2-7b-chat")
os.makedirs(llama_dir, exist_ok=True)
gguf_file = os.path.join(llama_dir, "llama-2-7b-chat.Q4_K_M.gguf")
if not os.path.exists(gguf_file):
    url = "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
    urlretrieve(url, gguf_file)

# Add placeholders for other models like ESRGAN, SwinIR, etc.
# For example, ESRGAN
esrgan_dir = os.path.join(MODEL_DIR, "restoration/esrgan")
os.makedirs(esrgan_dir, exist_ok=True)
# Assume download URL or skip if not specific

# Similarly for others

print("All models downloaded and saved locally.")