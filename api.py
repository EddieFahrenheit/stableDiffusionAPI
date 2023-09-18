from auth_token import auth_token
from diffusers import DiffusionPipeline
from fastapi import FastAPI, Response
import torch
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import base64 

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_credentials=True, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

device="mps"
model_id="runwayml/stable-diffusion-v1-5"
pipe = DiffusionPipeline.from_pretrained(model_id, use_auth_token=auth_token)
pipe.safety_checker = lambda images, clip_input: (images, False) 
# Recommended if your computer has < 64 GB of RAM
pipe = pipe.to(device)
pipe.enable_attention_slicing()

@app.get("/")
def generate(prompt: str):
    
    image = pipe(prompt, guidance_scale=8.5).images[0]

    image.save("testimage.png")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    imgstr = base64.b64encode(buffer.getvalue())
    
    return Response(content=imgstr, media_type="image/png")